# Milestone M12 report: A1' (maximum over the frequencies in the windowed-gain penalty) and A2' (two-timescale law with a persistent latent)

Contract: `docs/m12_contract.md` (D134-D136). Every agent writes its own section; the main session assembles the
verdicts and the decision log.

## Verdicts (main session, 21 October 2026)

Both flaws diagnosed in M11 are repaired and both repairs did what they were built to do; neither moves its
hypothesis into the confirmed branch, and the first one turns a corridor result into its opposite. The manuscript
(paper/v2, submitted 9 October) is not touched; the four new laws are further rows of the corridor tables (design
keys `wmax_laws`, `m12_laws`; not in the correlations nor the verdicts of H12); the manuscript's tables stay those of
the tag v2.0.2.

| Item | Rule (contract) | Numbers | Verdict |
|---|---|---|---|
| H-N3' windowed-gain penalty with the maximum over the frequencies (D134, section A') | GRU and LSTM stable share >= 0.9 at <= +10 % RMSE, long-window check holds | GRU (weight 100) 1.00 [1.00, 1.00] at +46.1 % [+43.3, +48.8]; LSTM (weight 10) 0.99 [0.97, 1.00] at +29.6 % [+27.9, +31.4]; the long-window check holds for both, every pole inside; PERL 0.04 at +34.4 % | **partial** by the rule: the maximum fixes the diagnosed blindness (M11: 0.09 and 0.66 stable), the linearised string stability of a recurrent law can now be enforced at every equilibrium, but at three to five times the accuracy cost the rule allows. In the corridor the fully stabilised laws collapse: gru_wmax 0.597 / 1.118 against gru_windowed 0.132 / 0.260 (+0.465 [+0.395, +0.541] on I-80, +0.859 [+0.710, +1.023] on US-101) with about 6 400 contacts per 1 000 vehicle-km and 61-79 % of the vehicles inserted; lstm_wmax 0.481 / 0.926 (no difference on I-80, +0.247 [+0.142, +0.370] on US-101); only PERL improves on its M11 version (-0.11 / -0.12) and still collides. The fine-tuned members are more stable than before (unstable share among equilibria 0.12-0.23), so the collapse is not a failure of the penalty but of the laws it produces: forcing every equilibrium of a recurrent law to be string stable at this price makes it sluggish at the insertion and the merges, where the corridor error is made. The corridor optimum of the penalty lies at moderate weights (M11: GRU 0.1, stable 0.09, the best GRU law), and the fold-0 GRU at weight 1 (+4.5 %, 0.82 stable) marks the region to search |
| H-N2' two-timescale law with a persistent latent (D135, section B') | RMSE <= 1.05 x GRU, 100 % stable equilibria at every frozen z and online, certificate at all 21 z of [-0.2, 1] after fine-tuning | RMSE 4.322 m = 2.071 x GRU (M11: 2.02 x; 0.958 x E4, 1.025 x tt); no unstable equilibrium at any frozen z or online on every fold, before and after fine-tuning; the exact certificate holds at all 21 z for all 5 runs (smallest margin 0.0050 s^-2 at z = -0.2); the latent now spans its range (99 % quantile 0.555 on the 15-s highD events, 0.959 on I-80); corridor: best of 27 laws on US-101 (0.074 against E 0.113, -0.039 [-0.047, -0.031]; against tt -0.016 [-0.024, -0.010]), no difference on I-80 (0.127, rank 5 of 32), contacts on US-101 0.49 against 0.23 (+0.26 [+0.02, +0.54]) | **refuted** by the accuracy criterion with both stability conditions met. This closes H-N2: with the latent free to move over the whole certified range the gap to the GRU does not shrink at all, so the accuracy of the recurrent laws on 15-s events is not a slow adaptation of the time gap; it lives in the fast, within-window response. What the slow latent buys is corridor fidelity out of sample: the certified structure with a persistent latent is the best law of the study on US-101 (the manuscript's hybrid: 0.113) at the price of a few more contacts |

Two conclusions across M11 and M12. (1) For recurrent laws the stability penalties move the law along one curve:
free (accurate, 90 % unstable, collides), moderately penalised (M11 windowed at weight 0.1: 9 % stable, the best
GRU in the corridor), fully stabilised (M12 maximum at weight 100: 100 % stable, +46 % RMSE, collapses in the
corridor); the corridor optimum is in the middle, which agrees with H-N1 (no linearised statistic predicts the corridor
error better than the pre-specified share) and with the manuscript's finding that fidelity comes from the certified
structure rather than from a penalty. (2) The certified hybrid remains the way to corridor fidelity, and the persistent
slow latent improves it out of sample without touching the certificate.

Next steps, in the order of cost: (a) the maximum-reduced penalty at moderate weights (0.3, 1, 3) on five folds with
the corridor (one pilot of 9 runs plus 15 fold runs and 90 corridor runs) to locate the optimum between the M11 and
M12 laws; (b) the safety filter with the leader's acceleration in the barrier (accepted item 2); (c) the nonlinear
explanation of the ablation (accepted item 3); (d) the slow latent carried along the event in training on the
longer NGSIM events only, where it can reach its range (the highD events cap it at 0.555).

Deviations from the contract, all recorded by the agents in their sections and in D134-D136: none of substance in
A' (the weights chosen by the D85 ties, a renamed statistic column in the full-history table); in B' the teacher-forced
latent pass was implemented as a fixed-size module (`cf_stability/train/stateful.py`) compatible with the CUDA graph,
so the fallback of D136 was not needed; the law file of a stateful model carries `stateful: true` in its hash. Agent
B' found and fixed a defect of the public release v2.1.0 (the design key `filter_laws` was not accepted by the
corridor tables; hotfix 990363b2 on the public main, the same change on main).

## A': windowed-gain penalty with the maximum over the frequencies (D134, H-N3')

Status: complete (2026-10-21 02:52). Every step of D134 was run: implementation and tests, pilot, choice of the weights,
five folds, full-history audit, long-window check, fine-tuning on NGSIM I-80, law files, 180 corridor runs, metrics,
asymmetry, tables and the paired tables. **H-N3': the numbers fall in the branch "partial" of the decision rule** (A'.5);
the verdict is left to the main session.

Summary: with the maximum over the frequencies the windowed-gain penalty reaches what the mean did not: at the weights
chosen by D85 (GRU 100, LSTM 10) the five-fold stable band share on highD is 1.00 [1.00, 1.00] (GRU) and 0.99 [0.97,
1.00] (LSTM), every pole inside, the long-window check holds; the price is the accuracy, +46.1 % [+43.3, +48.8] and
+29.6 % [+27.9, +31.4] test spacing RMSE against E1 (above the 10 % of the rule: branch "partial"). PERL is not repaired
(0.04 at weight 100). The share of speeds with a windowed gain above 1.02 at 0.5-2 rad/s goes from 0.19 to 0.20 (GRU,
whose remaining linear excess at 2 rad/s the rollouts do not see) and from 0.45 to 0.00 (LSTM). In the corridor the
members fine-tuned on I-80 keep most of that stability (unstable among equilibria 0.12 and 0.23), but the laws are worse
than those of the mean: gru_wmax 0.597 / 1.118 against gru_windowed 0.132 / 0.260 with about 6 400 contacts per 1 000
vehicle-km (against 270-298), lstm_wmax 0.481 / 0.926 against 0.423 / 0.679 (A'.7).

### A'.1 What was implemented

- **Key `train.penalty.frequency_reduce`** (`configs/train.yaml`, `PenaltyConfig` of
  `cf_stability/stability/penalties.py`): `mean` (the default, the penalty of M11, D128) or `max`. Refused with any
  other value and, away from its default, for every kind other than `windowed_gain` (as the two margins of D128).
  Hash-neutral at its default: one entry appended to `LATE_KEYS` of `cf_stability/utils.py`
  (`("train", "penalty", "frequency_reduce"): "mean"`).
- **The reduction** (`reduce_frequencies`, used by `windowed_terms`): the per-frequency terms relu(|G_w| - 1 + 0.02)^2
  at the 25 audit frequencies of every anchored equilibrium are reduced with `mean(dim=1)` (the M11 operation, unchanged
  bit for bit) or `amax(dim=1)` (the largest term; its gradient reaches the frequency of that term only). The term
  relu(0.5 - M_w), the pole term relu(rho - 0.995), the mean over the anchored equilibria and the existence term are
  unchanged.
- **History info** `argmax_omega` (appended to `INFO_KEYS["windowed_gain"]`, recorded under both reductions): the
  frequency in rad/s of the largest excess |G_w| - 1 + 0.02 over the anchored equilibria and the audit frequencies (the
  frequency of `max_gain`; NaN without an anchored equilibrium). In the history: `penalty_argmax_omega` (the mean over
  the penalty steps of the epoch, as for every info entry) and `val_penalty_argmax_omega` (the validation penalty at the
  grid speeds after the epoch: one audit frequency).
- **Queues** `configs/queue/e2_wmax_pilot.yaml` (12 jobs: GRU, LSTM, PERL; weights 0.1, 1, 10, 100; follownet_highd
  fold 0, seed 0; `every` 4, 16 speeds; steps train, audit, platoon; experiments `e2_wmax_w<weight>`),
  `configs/queue/e2_wmax.yaml` (15 jobs at the chosen weights, folds 0-4, the fold-0 jobs are pilot runs) and
  `configs/queue/m12_ft_wmax.yaml` (15 jobs on ngsim_i80, `init_from` the e2_wmax run of the same fold, the penalty
  continued with the same kind, reduction, weight, `every` and speeds; steps train and audit; experiments
  `m12_ft_wmax_w<weight>`), as `m11_ft_windowed`. `tests/test_run_experiment.py`: the three files and their assertions
  appended.
- **Laws** `gru_wmax`, `lstm_wmax`, `perl_wmax` appended to `configs/corridor/laws.yaml` (kind models, cuda, members
  `runs/m12_ft_wmax_w{100,10,100}/ngsim_i80/{gru,lstm,perl}/driver_fold{fold}_seed0`).
- **Tables of the pilot and the five folds** (`scripts/analysis/windowed_gain.py --arm wmax`; the default `--arm
  windowed` writes the M11 tables unchanged): `runs/_tables/m12/e2_wmax_pilot` (weights 0.1-100, with the largest
  windowed gain of the validation penalty at the best epoch and its frequency, `val_penalty_argmax_omega`),
  `e2_wmax_chosen.json` (D85, ties as in D128: `choose_weight` of `cf_stability/eval/tables.py`), `e2_wmax` (per
  architecture E1, the penalty of M11 at the weights D85 chose for it - the rows "before": GRU 0.1, LSTM 10, PERL 1 -,
  the chosen weight of the maximum; the fine-tuned rows on NGSIM I-80 against `m5_ft`) and `e2_wmax_branch.md` (the
  branch of the rule of H-N3', computed from the chosen weight of the maximum only), `missing_wmax.txt`.
- **Full-history audit** `scripts/analysis/full_history_audit.py --arm-set m12` (`M12_ARMS`: E1 seed 0, the chosen
  weights of M11 and of M12 on five folds, the pilot weights of both on fold 0) into `runs/_tables/m12/
  full_history_wmax.{csv,md}`, with one statistic more for this arm set only (`M12_STATISTICS`, `high_band_unstable`):
  the share of the speeds whose windowed gain exceeds 1.02 at an audit frequency in 0.5-2 rad/s (the 8 audit
  frequencies 0.52-2 rad/s), the secondary number of D134. `full_history.json` is written into `e2_wmax*` runs only.
- **Long-window check** `scripts/analysis/long_window_check.py --arm-set m12` (`M12_ARMS`: E1, the M11 pilot weights,
  the M12 pilot weights, fold 0, seed 0) into `runs/_tables/m12/long_window_wmax.{csv,md}` and
  `long_window_wmax_runs.json`; `long_window.json` is written into `e2_wmax*` runs only.
- **Corridor** (prepared before the runs): `cf_stability/eval/corridor_tables.py` `WMAX_LAWS` (`gru_wmax`,
  `lstm_wmax`, `perl_wmax`), `WMAX_TEXT`, the design field `wmax_laws`, `wmax_note()`, and the law set in
  `table_laws()`, `present()` (origin "a law of M12") and the instability rows, the note line in laws, components,
  instability, asymmetry and contacts_absolute - each next to the corresponding `windowed_laws` entry; the key
  `wmax_laws` of the design of `configs/corridor_metrics.yaml`; `wmax_laws` appended to `EXCLUDED_LAW_SETS` of
  `cf_stability/eval/figures_supplement.py`; `scripts/analysis/wmax_pairs.py` (the builder of `ablation_pairs` with
  the pairs gru_wmax - gru_windowed, gru_wmax - gru, lstm_wmax - lstm_windowed, lstm_wmax - lstm, perl_wmax -
  perl_windowed, perl_wmax - perl, into `runs/_tables/m12/wmax_pairs.{csv,md}`).

What the reduction changes on stored M11 models (validation penalty at the grid speeds, CPU, for illustration): GRU
weight 10 (the model of the diagnosis of D128, largest windowed gain 1.109 at 1.12 rad/s) gain term 0.0016 (mean) ->
0.0117 (max), penalty 0.0034 -> 0.0135, above the feasibility tolerance 0.01 of D89; LSTM weight 10 (largest gain
1.086 at 2 rad/s) 0.0002 -> 0.0037, penalty 0.0017 -> 0.0052, still below 0.01: the mean over the anchored
equilibria (unchanged by D134) still averages an amplification that only some speeds have.

### A'.2 Hash neutrality and reproducibility checks

- Dry runs of all 27 queue files (`python scripts/run_experiment.py configs/queue/<file>.yaml --dry-run`, 1 183 jobs)
  before the change (17:50-17:58) and after it (18:00-18:17): the outputs are byte-identical (`cmp` of the 27 files).
- `python scripts/make_tables.py` before the change (17:59) and after it (18:20): `runs/_tables/m4` and
  `runs/_tables/m8/e2_monotone` byte-identical to git both times.
- `python scripts/analysis/full_history_audit.py --no-json --no-figure` (m9, 207 runs) and `--arm-set m11 --no-json`
  (39 runs) with the extended script: `runs/_tables/m9/full_history.{csv,md}` and `runs/_tables/m11/
  full_history_windowed.{csv,md}` byte-identical to git (18:23-18:25, and again 23:12-23:13 with the final script).
- `python scripts/analysis/windowed_gain.py tables` before the refactoring (18:12) and after it (18:25 and 18:46, the
  final version): `runs/_tables/m11/e2_windowed_pilot`, `e2_windowed`, `e2_windowed_branch.md`,
  `e2_windowed_chosen.json`, `missing_windowed.txt` byte-identical to git.
- The training, audit and platoon code that the M12 runs use was changed by agent B' at the same time (the stateful
  interface of D135, additive: `state=None` keeps the earlier path). Check on the audit path: a copy of the stored run
  `e2_windowed_w10/follownet_highd/gru/driver_fold0_seed0` (in the scratchpad) audited again with the current code
  (`scripts/audit_stability.py run=<copy>`, 19:2x) gives an `audit` block identical to the stored one bit for bit
  (wall time and paths left out). The trainer changes of B' leave the draws and the calls of a model without a state
  unchanged (read in the diff; `seg_starts` drawn as before).
- The default reproduces the M11 penalty bit for bit on stored runs: on CUDA (the device of their training) the
  validation penalty of the best epoch recomputed from `model.pt` and the band of `metrics.json` (the trainer's
  `validation_penalty`) equals the value and every part stored in the history of `e2_windowed_w10` GRU and LSTM and
  `e2_windowed_w1` PERL (fold 0, seed 0), bit for bit, with the key left out and with `mean` given; on the CPU the
  default equals `mean` bit for bit and the stored CUDA values to rel 1e-4 (float32 model on another device).

### A'.3 Tests

`tests/test_penalties_windowed.py` (M11 tests of the kind rerun with both values, new tests for D134):
- parametrised over `mean` and `max`: the response of the linear laws with memory (three laws: the two of M11 and a
  resonant one), the penalty against the closed form, the memoryless limit, the central-difference gradients of each
  term and of all three, the agreement with `full_history_audit.analyse` on three stored runs (now also the largest
  windowed gain, its frequency and the per-frequency terms reduced), anchoring and existence, the training on synthetic
  data (with the frequency of the largest excess in the history), the CUDA test of the eager penalty steps;
- new: the key and its hash-neutral default (`test_frequency_reduce_key_and_its_hash_neutral_default`); where the
  gradient of each reduction goes (`test_reduce_frequencies_and_where_its_gradient_goes`); on the linear toy law (and
  on the resonant law) `max` equals the largest per-frequency term relu(|G_w| - 0.98)^2 of the closed form, the info
  names its frequency, and the gradient of the penalty with respect to the history Jacobian is that of the term of this
  frequency alone (equal to rtol 1e-12; different from that of every other amplifying frequency), the gradient with
  respect to |G_w| is one-hot at it; on the resonant law (largest gain 1.116 near 0.36 rad/s) the mean is below 0.01
  and the max above it (`test_max_is_the_largest_per_frequency_term_and_its_gradient_reaches_that_frequency_only`); the
  training with the key left out and with `mean` identical bit for bit, `max` different
  (`test_the_default_trains_as_the_explicit_mean`); the stored M11 runs (`test_the_default_reproduces_the_stored_m11_penalty`,
  CPU, and `..._bit_for_bit_on_cuda`, opt-in `CF_GPU_TESTS=1`).
- `tests/test_utils.py`: the new late key.
- `tests/test_windowed_analysis.py` (5 new tests): `high_band_unstable` on a hand-made payload; the arm set m12 of the
  full-history audit (its arms, the share of the band equal to that of the written files, the files written into
  `e2_wmax*` only); the arm set m12 of the long-window check (table `long_window_wmax`, files into `e2_wmax*` only);
  the branch of H-N3'; the tables of `--arm wmax` without runs and the frequency of the best epoch read from the
  history (the M11 tables keep their columns).
- `tests/test_corridor_tables_wmax.py` (4 tests): the rows of a present law of `wmax_laws` in laws, components and
  instability with its member audits and its own note line, not in the correlations; an absent one left out with a
  note; the design key and `EXCLUDED_LAW_SETS`; the paired table of `wmax_pairs.py` on the hand-made tree.

Results (CPU, 2026-10-20): `tests/test_windowed_analysis.py` 11 passed; `tests/test_corridor_tables_wmax.py
tests/test_corridor_tables_windowed.py` 5 passed, then the paired test 1 passed; the suites touched by the shared
edits, `tests/test_analysis_m9.py tests/test_figures_supplement.py tests/test_corridor_tables.py
tests/test_corridor_tables_ablation.py tests/test_corridor_tables_m8.py tests/test_corridor_tables_review.py`: all
passed (18:55-19:25); after the queue files and the laws were written, `tests/test_penalties_windowed.py
tests/test_penalties.py tests/test_utils.py tests/test_windowed_analysis.py tests/test_corridor_tables_wmax.py
tests/test_corridor_tables_windowed.py tests/test_run_experiment.py tests/test_trainer.py`: 151 passed, 29 skipped
(the opt-in GPU tests), 21:35-21:46 (with the trainer as changed by agent B').

Results (2026-10-20): `python -m pytest tests/test_penalties_windowed.py tests/test_utils.py tests/test_penalties.py`
passed (CPU; the GPU tests skipped); `CF_GPU_TESTS=1 python -m pytest tests/test_penalties_windowed.py -k "cuda or graph
or eager"`: 5 passed (the three stored runs bit for bit on CUDA and the graph test with both reductions).

### A'.4 Pilot (`configs/queue/e2_wmax_pilot.yaml`, 12 runs, follownet_highd fold 0, seed 0)

12 jobs done, none failed (2026-10-20 18:20-21:14, 2 workers; the queue waited five times for free memory, 1.9-4.8 GB <
6 GB, while other processes of the machine held it). Table `runs/_tables/m12/e2_wmax_pilot.md` (`python
scripts/analysis/windowed_gain.py choose --arm wmax`, 21:15; RMSE change against the E1 run of fold 0, seed 0, paired
over the 2 503 test events; stable = band share of the audit, speeds in support; the long-window columns are filled
by the later run of `tables`):

| arch | weight | val RMSE | test RMSE | change vs E1 | stable | unstable | outside | max gain (audit) | feasible epochs | val penalty | gain term (max) | min M_w | max rho | max windowed gain (val) at omega |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GRU | 0.1 | 2.34 | 2.27 | +10.2 % | 0.36 | 0.59 | 0.05 | 1.208 | 0 | 0.552 | 0.0085 | -0.004 | 0.998 | 1.210 at 0.05 |
| GRU | 1 | 2.22 | 2.16 | +4.5 % | 0.82 | 0.18 | 0.00 | 1.149 | 47 | 0.000 | 0.0002 | 0.783 | 0.995 | 0.996 at 0.02 |
| GRU | 10 | 3.16 | 3.14 | +52.2 % | 1.00 | 0.00 | 0.00 | 1.001 | 13 | 0.000 | 0.0004 | 1.149 | 0.995 | 1.008 at 2.0 |
| GRU | 100 | 2.96 | 2.97 | +43.7 % | 1.00 | 0.00 | 0.00 | 1.012 | 9 | 0.004 | 0.0041 | 0.758 | 0.995 | 1.075 at 2.0 |
| LSTM | 0.1 | 2.34 | 2.34 | +4.9 % | 0.41 | 0.55 | 0.05 | 1.572 | 0 | 0.314 | 0.0233 | -0.019 | 0.998 | 1.599 at 0.24 |
| LSTM | 1 | 2.93 | 2.90 | +30.1 % | 1.00 | 0.00 | 0.00 | 0.998 | 2 | 0.008 | 0.0003 | 0.461 | 0.992 | 0.998 at 0.02 |
| LSTM | 10 | 2.92 | 2.89 | +29.7 % | 0.95 | 0.00 | 0.05 | 0.998 | 9 | 0.007 | 0.0002 | 0.680 | 0.996 | 0.998 at 0.02 |
| LSTM | 100 | 3.10 | 3.07 | +38.0 % | 0.09 | 0.91 | 0.00 | 1.050 | 1 | 0.006 | 0.0059 | 0.511 | 0.987 | 1.065 at 2.0 |
| PERL | 0.1 | 3.13 | 3.07 | +26.1 % | 0.00 | 0.91 | 0.09 | 1.141 | 0 | 0.581 | 0.0141 | -0.000 | 1.000 | 1.215 at 0.06 |
| PERL | 1 | 3.24 | 3.18 | +30.5 % | 0.00 | 1.00 | 0.00 | 3.847 | 0 | 0.512 | 0.0005 | -0.001 | 1.002 | 1.006 at 0.08 |
| PERL | 10 | 3.32 | 3.28 | +34.5 % | 0.00 | 0.95 | 0.05 | 1.134 | 0 | 0.537 | 0.0174 | 0.000 | 0.999 | 1.139 at 0.08 |
| PERL | 100 | 3.16 | 3.07 | +25.8 % | 0.05 | 0.77 | 0.18 | 6.377 | 0 | 1.388 | 0.8485 | -0.000 | 1.124 | 4.622 at 0.63 |

(val penalty, gain term, min M_w, max rho, largest windowed gain and its frequency: the validation penalty of the best
epoch at the grid speeds; the gain term is now the mean over the anchored equilibria of the maximum over the
frequencies.)

**Chosen weights (rule of D85, `runs/_tables/m12/e2_wmax_chosen.json`, written into `configs/queue/e2_wmax.yaml`):**
GRU 100 (stable 1.00 at weights 10 and 100: the smaller validation RMSE, 2.96 against 3.16), LSTM 10 (weights 1 and 10
reach 0.9, stable 1.00 and 0.95: the smaller validation RMSE, 2.92 against 2.93), PERL 100 (no weight reaches 0.9: the
largest stable share, 0.05; no tie).

Observations (no verdict):
- The maximum makes the penalty bite where the mean did not: GRU and LSTM reach a stable band share of 0.95-1.00 at
  weights 10 and 100 (GRU) and 1 and 10 (LSTM), against at most 0.23 (GRU) and 0.45 (LSTM) with the mean in M11 at
  weights 0.1-10; the price is the accuracy, +30 % to +52 % test RMSE on fold 0 at these weights.
- GRU at weight 1 is the closest to the region of H-N3' (+4.5 %, stable 0.82, 47 feasible epochs, validation penalty
  0.0003): its windowed linearisation is stable at every grid speed (largest windowed gain 0.996), but the audit's
  rollouts amplify an oscillation of the leader speed of 0.2 m/s by 1.08-1.15 at 8-11 m/s, at 1.36-1.65 rad/s, where
  the fit of the response has relative residuals of 0.08-0.11 (at 2 rad/s 0.93): a response that is not the linear
  one, which the analytic penalty cannot see (it linearises at the constant equilibrium history). D85 does not choose
  this weight (share < 0.9).
- LSTM at weight 100 is worse than at 10 (stable 0.09, every unstable speed at about 1.05 at 2 rad/s, one feasible
  epoch); GRU at 100 has its largest windowed gain of the validation penalty at 1.075 (2 rad/s) but every audited speed
  stable (largest numerical gain 1.012, below the threshold 1.02 of the audit): the weight does not act monotonically.
- PERL: at no weight a stable equilibrium in the band (0.05 at weight 100); the term relu(0.5 - M_w) stays near 0.5
  (min M_w about 0, as with the mean) and at weight 100 the poles leave the unit circle (rho 1.12) and the gains reach
  6.4: the penalty does not repair PERL.
- The frequency of the largest excess at the best epoch (`val_penalty_argmax_omega`) moves from the low frequencies
  (0.02-0.24 rad/s at weight 0.1) to 2 rad/s (GRU 10 and 100, LSTM 100), the upper end of the audit grid, or to 0.02
  rad/s with every gain below 1 (GRU 1, LSTM 1 and 10).

Durations: 10.1-48.4 min per job (training 8.4-44.6 min, 13-94 epochs of 222 steps; audit and platoon 1-3 min).

### A'.5 Five folds (`configs/queue/e2_wmax.yaml`) and the branch of the rule of H-N3'

Queue with the chosen weights (GRU 100, LSTM 10, PERL 100; `every` 4, 16 speeds): 15 jobs, of which the 3 of fold 0
are the pilot runs (skipped); 12 runs trained, audited and platoon-tested on 2026-10-20 21:16-23:08 with 2 workers,
none failed (9.4-32.7 min per job). Table `runs/_tables/m12/e2_wmax.md` (`python scripts/analysis/windowed_gain.py
tables --arm wmax`, 23:09; follownet_highd, seed 0, five folds; RMSE change against the E1 runs of the same folds and
seed, paired over the 12 512 test events; stable = mean over the runs of the stable band share with the interval over
the runs; the rows "mean (M11)" are those of `runs/_tables/m11/e2_windowed.md`, recomputed):

| arch | arm | test RMSE (m) | change vs E1 | stable | unstable | not stable (poles incl.) | max gain (median) | long window (fold 0) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| GRU | E1 | 2.09 [2.05, 2.13] | | 0.09 [0.00, 0.18] | 0.50 | 0.91 | 2.577 | fails |
| GRU | mean (M11), weight 0.1 | 2.45 [2.41, 2.48] | +17.3 % [+15.1, +19.2] | 0.09 [0.04, 0.16] | 0.91 | 0.91 | 1.370 | fails |
| GRU | **max, weight 100** | 3.05 [3.00, 3.09] | +46.1 % [+43.3, +48.8] | 1.00 [1.00, 1.00] | 0.00 | 0.00 | 0.997 | holds |
| LSTM | E1 | 2.18 [2.15, 2.21] | | 0.10 [0.05, 0.15] | 0.36 | 0.90 | 133.894 | fails |
| LSTM | mean (M11), weight 10 | 2.97 [2.92, 3.01] | +36.2 % [+34.2, +38.3] | 0.66 [0.35, 0.97] | 0.34 | 0.34 | 1.081 | holds |
| LSTM | **max, weight 10** | 2.82 [2.78, 2.87] | +29.6 % [+27.9, +31.4] | 0.99 [0.97, 1.00] | 0.00 | 0.01 | 0.999 | holds |
| PERL | E1 | 2.37 [2.33, 2.41] | | 0.06 [0.03, 0.11] | 0.80 | 0.94 | 39.321 | fails |
| PERL | mean (M11), weight 1 | 3.13 [3.08, 3.18] | +31.9 % [+30.0, +33.7] | 0.00 [0.00, 0.00] | 0.99 | 1.00 | 1.429 | fails |
| PERL | **max, weight 100** | 3.19 [3.13, 3.24] | +34.4 % [+32.3, +36.4] | 0.04 [0.00, 0.08] | 0.87 | 0.96 | 1.305 | fails |

Per fold (stable share; test RMSE; largest numerical gain): GRU 1.00 at every fold (2.97-3.14 m; 0.997-1.012); LSTM
0.95, 1.00, 1.00, 1.00, 1.00 (2.29-3.00 m; fold 4 trained 100 epochs, the limit, test RMSE 2.29 m; 0.998-0.999); PERL
0.05, 0.00, 0.00, 0.14, 0.00 (3.02-3.53 m; largest gains 1.13-15.7). Every pole of the full-history loop inside the unit
circle at the audited speeds of GRU and LSTM (A'.6).

**Branch of the decision rule of H-N3'** (`runs/_tables/m12/e2_wmax_branch.md`): GRU stable share 1.000 (>= 0.9), RMSE
change +46.1 % (> 10 %), long-window check holds; LSTM 0.991 (>= 0.9), +29.6 % (> 10 %), holds; PERL (not judged)
0.036, +34.4 %, fails. Both judged architectures reach the share, both at an increase above 10 %: **the numbers fall in
the branch "partial"** (the share is reached at a larger increase). No verdict here.

### A'.6 Long-window check and full-history audit

**Long-window check** (`python scripts/analysis/long_window_check.py --arm-set m12 --device cuda`, 2026-10-20
21:19-21:33, 24 runs; table `runs/_tables/m12/long_window_wmax.md`, per-equilibrium results in
`long_window_wmax_runs.json`, `long_window.json` written into the 12 pilot runs of `e2_wmax*` only; the rows of E1 and
of the M11 pilot recomputed without writing, equal to those of `runs/_tables/m11/long_window.md`): the rollout gain
over the last 252 s of a 380-s rollout at 0.02, 0.05 and 0.1 rad/s at the audit equilibria inside the band at the
speeds in support, fold 0, seed 0.

| arch | arm | speeds | max gain 0.02 / 0.05 / 0.1 rad/s | share > 1.02 | holds |
|---|---|---:|---|---:|---|
| GRU | E1 | 4 | 1.153 / 1.214 / 1.199 | 1.00 | no |
| GRU | mean w0.1 (M11, chosen) | 22 | 1.125 / 1.339 / 1.310 | 0.64 | no |
| GRU | max w0.1 | 21 | 1.137 / 1.207 / 1.177 | 0.52 | no |
| GRU | max w1 | 22 | 0.996 / 0.986 / 0.978 | 0.00 | yes |
| GRU | max w10 | 22 | 0.998 / 0.992 / 0.987 | 0.00 | yes |
| GRU | **max w100 (chosen)** | 22 | 0.997 / 0.989 / 0.982 | 0.00 | yes |
| LSTM | E1 | 12 | 1.760 / 1.877 / 1.155 | 0.50 | no |
| LSTM | mean w10 (M11, chosen) | 22 | 0.995 / 0.985 / 0.980 | 0.00 | yes |
| LSTM | max w0.1 | 21 | 1.276 / 1.430 / 1.507 | 0.29 | no |
| LSTM | max w1 | 22 | 0.998 / 0.990 / 0.973 | 0.00 | yes |
| LSTM | **max w10 (chosen)** | 21 | 0.998 / 0.990 / 0.977 | 0.00 | yes |
| LSTM | max w100 | 22 | 0.999 / 0.996 / 0.988 | 0.00 | yes |
| PERL | E1 | 19 | 1.106 / 1.333 / 2.915 | 0.79 | no |
| PERL | mean w1 (M11, chosen) | 22 | 1.015 / 1.731 / 2.960 | 0.59 | no |
| PERL | max w0.1 | 20 | 1.000 / 1.036 / 1.042 | 0.90 | no |
| PERL | max w1 | 22 | 1.463 / 1.432 / 4.673 | 1.00 | no |
| PERL | max w10 | 21 | 0.999 / 0.990 / 1.139 | 0.52 | no |
| PERL | **max w100 (chosen)** | 18 | 1.006 / 6.153 / 1.038 | 0.56 | no |

At the chosen weights the check holds for GRU (100) and LSTM (10) and fails for PERL (100). For GRU and LSTM it holds
at every weight from 1 up (the mean of M11: GRU 1 and 10, LSTM 10). LSTM at weight 100 holds although its audit finds
91 % of the speeds unstable: its amplification (about 1.05) sits at 2 rad/s, above the frequencies of this check.

**Full-history audit** (`python scripts/analysis/full_history_audit.py --arm-set m12`, 23:08 and again at 23:10 after
a fix of the column name of the new statistic, see A'.9; 66 runs: E1 seed 0, the chosen weights of M11 and of M12 on
five folds, the pilot weights of both on fold 0; table `runs/_tables/m12/full_history_wmax.md`, speeds in support,
mean over the runs with the interval over the runs; `full_history.json` written into the 24 runs of `e2_wmax*`):

| arch | arm | runs | speeds | poles inside | windowed > 1.02 | numerical unstable | agreement | windowed > 1.02 at 0.5-2 rad/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | E1 | 5 | 104 | 1.00 | 0.89 [0.79, 0.99] | 0.89 | 1.00 | 0.51 [0.29, 0.74] |
| GRU | mean (M11), 0.1 | 5 | 110 | 1.00 | 0.91 [0.84, 0.96] | 0.91 | 1.00 | 0.19 [0.05, 0.33] |
| GRU | max, 100 | 5 | 110 | 1.00 | 0.20 [0.00, 0.56] | 0.00 | 0.80 [0.44, 1.00] | 0.20 [0.00, 0.56] |
| LSTM | E1 | 5 | 101 | 0.94 | 0.81 [0.75, 0.88] | 0.84 | 0.97 | 0.37 [0.17, 0.63] |
| LSTM | mean (M11), 10 | 5 | 110 | 1.00 | 0.45 [0.14, 0.77] | 0.34 | 0.89 | 0.45 [0.14, 0.77] |
| LSTM | max, 10 | 5 | 110 | 1.00 | 0.00 [0.00, 0.00] | 0.00 | 1.00 | 0.00 [0.00, 0.00] |
| PERL | E1 | 5 | 106 | 0.94 | 0.93 [0.87, 0.97] | 0.93 | 1.00 | 0.46 [0.30, 0.62] |
| PERL | mean (M11), 1 | 5 | 110 | 0.70 [0.30, 1.00] | 0.90 [0.72, 1.00] | 1.00 | 0.90 | 0.82 [0.58, 1.00] |
| PERL | max, 100 | 5 | 109 | 0.70 [0.30, 1.00] | 0.65 [0.33, 0.93] | 0.96 | 0.69 [0.34, 0.95] | 0.51 [0.15, 0.87] |

**Secondary number of D134** (no verdict): the share of the speeds in support whose largest windowed gain at 0.5-2 rad/s
exceeds 1.02 is, before (the mean, M11, chosen weights) and after the change (the maximum, chosen weights), GRU 0.19
[0.05, 0.33] -> 0.20 [0.00, 0.56], LSTM 0.45 [0.14, 0.77] -> 0.00 [0.00, 0.00], PERL 0.82 [0.58, 1.00] -> 0.51 [0.15,
0.87] (E1: 0.51, 0.37, 0.46). Over the pilot weights of fold 0 the share of GRU is 0.00, 0.82, 1.00 with the mean at
0.1, 1, 10 and 0.09, 0.00, 0.00, 0.91 with the maximum at 0.1, 1, 10, 100; of LSTM 0.27, 0.41, 0.73 (mean) and 0.23,
0.00, 0.00, 1.00 (maximum). The GRU at weight 100 keeps windowed gains above 1.02 at 0.5-2 rad/s at a fifth of its
speeds (largest at 2 rad/s, the upper end of the grid) that the numerical audit does not find (its largest gains
0.997-1.012): the linearised and the numerical verdicts agree at 0.80 of the speeds, against 1.00 for the mean (M11)
and for LSTM with the maximum. The low-frequency coefficient decides the side of 1 at 0.02 rad/s at every speed of GRU
and LSTM with the maximum; sign(M) = sign(M_w) at all of their speeds (M_w >= 0.5 makes both positive).

### A'.7 Fine-tuning on NGSIM I-80 (`configs/queue/m12_ft_wmax.yaml`) and the corridor

**Fine-tuning** (15 runs, 2026-10-20 23:08 - 2026-10-21 00:39 with 2 workers, 4.6-19.4 min per job, none failed): the
e2_wmax runs of the same fold, the penalty continued (same kind, reduction, weight, `every`, speeds; the band of the
NGSIM data), steps train and audit. Rows of `runs/_tables/m12/e2_wmax.md` (ngsim_i80, five folds, seed 0; RMSE change
against `m5_ft`, the E1 models fine-tuned in M5, paired over the 5 521 test drivers; for information, the members of
the corridor laws):

| arch | arm | test RMSE (m) | change vs m5_ft | stable | unstable | max gain (median) |
|---|---|---:|---:|---:|---:|---:|
| GRU | m5_ft | 4.53 [4.45, 4.62] | | 0.27 [0.12, 0.43] | 0.64 | 1694 |
| GRU | mean (M11) 0.1, fine-tuned | 4.59 [4.49, 4.70] | +1.5 % [+0.0, +3.1] | 0.63 [0.50, 0.76] | 0.37 | 1.141 |
| GRU | **max 100, fine-tuned** | 5.07 [4.97, 5.18] | +11.9 % [+10.4, +13.6] | 0.88 [0.71, 0.99] | 0.11 | 1.043 |
| LSTM | m5_ft | 4.72 [4.62, 4.83] | | 0.25 [0.06, 0.45] | 0.67 | 861 |
| LSTM | mean (M11) 10, fine-tuned | 4.97 [4.86, 5.10] | +5.2 % [+3.5, +7.1] | 0.91 [0.78, 1.00] | 0.09 | 1.052 |
| LSTM | **max 10, fine-tuned** | 5.10 [4.99, 5.23] | +8.1 % [+6.3, +10.1] | 0.77 [0.40, 1.00] | 0.23 | 0.998 |
| PERL | m5_ft | 6.47 [5.87, 7.50] | | 0.06 [0.01, 0.11] | 0.81 | 785 |
| PERL | mean (M11) 1, fine-tuned | 8.38 [7.72, 9.38] | +29.4 % [+23.2, +36.6] | 0.17 [0.00, 0.43] | 0.83 | 1.500 |
| PERL | **max 100, fine-tuned** | 6.19 [6.02, 6.37] | -4.4 % [-17.4, +5.6] | 0.85 [0.69, 0.95] | 0.15 | 1.063 |

Per fold (stable share; test RMSE): GRU 0.93, 0.93, 0.54, 1.00, 1.00 (4.36-5.19 m); LSTM 0.00, 0.86, 1.00, 1.00, 1.00
(4.36-5.27 m; fold 0 unstable at every speed in support, largest gain 2.05, 13 epochs); PERL 0.93, 1.00, 0.54, 0.85, 0.92
(4.98-6.71 m). The fine-tuning keeps most of the stability of the highD members for GRU, loses it on one fold of the
LSTM, and gives PERL, unstable on highD, a stable share of 0.85 on the NGSIM band (with 0-6 feasible epochs per run;
the RMSE change of PERL against `m5_ft` is dominated by the 15.9 m of `m5_ft` fold 3). Information on the corridor
members, not part of the rule of H-N3' (highD).

**Law files** (00:39, `export_laws(..., only=["gru_wmax", "lstm_wmax", "perl_wmax"])`, so that the law files of the
other agent's corridor runs were not rewritten): `gru_wmax`, `lstm_wmax`, `perl_wmax`, 5/5 members each, window 30,
support 0.0-18.1 m/s, cuda. `python scripts/check_environment.py` (00:39): 17 checks, 17 OK.

**Corridor runs**: 180 runs (3 laws x 6 scenarios x 10 seeds, workers 3; the GPU and the CPU shared with agent B' and
other sessions), 2026-10-21 00:39-02:48, none failed (a second pass found all 180 complete); 106-159 s per run (GRU and
LSTM about 119 s, PERL about 145 s). `corridor_metrics.py law=<law>` and `corridor_asymmetry.py law=<law> workers=2` for
the three laws: 180 macro.json and 180 asymmetry.json, all OK. Then `python scripts/make_tables.py` (`runs/_tables/m4`
and `m8/e2_monotone` unchanged) and `python scripts/corridor_metrics.py compute=false tables=true` (02:50-02:51): the
three laws are further rows of `runs/_tables/m5/laws`, `components`, `instability` and of `runs/_tables/m8/asymmetry`,
`contacts_absolute` (1 910 runs in the tables); `instability_correlation`, `correlation_pooled`,
`correlation_clustered`, `verdicts`, `h12_1`, `h12_2`, `tost` (csv) are unchanged, their Markdown only in the run count
of the header line.

| corridor | law | macro error | dynamic macro error | contacts / 1000 veh-km | inserted | members: unstable among eq. | asymmetry index |
|---|---|---:|---:|---:|---:|---:|---:|
| I-80 | gru | 0.192 [0.164, 0.223] | 0.312 | 206 | 0.989 | 0.715 | |
| I-80 | gru_windowed (M11) | 0.132 [0.112, 0.153] | 0.233 | 270 | 1.000 | 0.370 | 0.623 |
| I-80 | **gru_wmax** | 0.597 [0.526, 0.679] | 0.468 | 6 384 | 0.786 | 0.121 | 0.640 |
| I-80 | lstm | 0.716 [0.662, 0.774] | 0.657 | 4 664 | 0.335 | 0.733 | |
| I-80 | lstm_windowed (M11) | 0.423 [0.385, 0.465] | 0.511 | 2 666 | 0.950 | 0.092 | 0.450 |
| I-80 | **lstm_wmax** | 0.481 [0.421, 0.542] | 0.412 | 3 386 | 0.867 | 0.229 | 0.462 |
| I-80 | perl | 0.431 [0.396, 0.464] | 0.652 | 11 683 | 0.989 | 0.940 | |
| I-80 | perl_windowed (M11) | 0.574 [0.544, 0.611] | 0.471 | 12 560 | 0.849 | 0.829 | 0.612 |
| I-80 | **perl_wmax** | 0.464 [0.430, 0.501] | 0.471 | 12 584 | 0.931 | 0.153 | 0.615 |
| US-101 | gru | 0.421 [0.338, 0.515] | 0.517 | 211 | 0.909 | 0.715 | |
| US-101 | gru_windowed (M11) | 0.260 [0.222, 0.304] | 0.481 | 298 | 1.000 | 0.370 | 0.390 |
| US-101 | **gru_wmax** | 1.118 [0.944, 1.314] | 0.975 | 6 463 | 0.612 | 0.121 | 0.635 |
| US-101 | lstm | 0.888 [0.784, 1.004] | 0.837 | 2 858 | 0.382 | 0.733 | |
| US-101 | lstm_windowed (M11) | 0.679 [0.607, 0.762] | 0.654 | 2 836 | 0.781 | 0.092 | 0.485 |
| US-101 | **lstm_wmax** | 0.926 [0.785, 1.086] | 0.766 | 4 108 | 0.674 | 0.229 | 0.500 |
| US-101 | perl | 0.661 [0.621, 0.704] | 0.685 | 13 481 | 0.891 | 0.940 | |
| US-101 | perl_windowed (M11) | 0.904 [0.821, 0.997] | 0.767 | 12 350 | 0.706 | 0.829 | 0.622 |
| US-101 | **perl_wmax** | 0.789 [0.695, 0.916] | 0.726 | 12 196 | 0.788 | 0.153 | 0.609 |

(macro error with its interval over the 30 runs of the law on the corridor; contacts: mean over the runs; inserted:
share of the planned vehicles inserted; members: mean over the five fold members of their audit share of unstable
equilibria; asymmetry: the acceleration asymmetry index of D121, ground truths 0.94-1.03; the rows of the free laws and
of M11 as in `runs/_tables/m5/laws.md`.)

**Paired comparisons** (`python scripts/analysis/wmax_pairs.py`, 02:51; table `runs/_tables/m12/wmax_pairs.md`; paired
by scenario and seed, 30 pairs per corridor and pair, difference candidate - reference with its percentile bootstrap
interval; descriptive, no verdict):

| corridor | pair | macro error | difference [95 %] | contacts / 1000 veh-km: difference [95 %] |
|---|---|---|---|---|
| I-80 | gru_wmax - gru_windowed | 0.597 vs 0.132 | +0.465 [+0.395, +0.541] | +6 113 [+5 490, +6 723] |
| I-80 | gru_wmax - gru | 0.597 vs 0.192 | +0.405 [+0.344, +0.472] | +6 178 [+5 572, +6 775] |
| I-80 | lstm_wmax - lstm_windowed | 0.481 vs 0.423 | +0.058 [-0.018, +0.131] | +721 [+328, +1 149] |
| I-80 | lstm_wmax - lstm | 0.481 vs 0.716 | -0.234 [-0.314, -0.158] | -1 278 [-2 836, +75] |
| I-80 | perl_wmax - perl_windowed | 0.464 vs 0.574 | -0.110 [-0.142, -0.078] | +24 [-521, +737] |
| I-80 | perl_wmax - perl | 0.464 vs 0.431 | +0.033 [-0.007, +0.074] | +900 [+269, +1 687] |
| US-101 | gru_wmax - gru_windowed | 1.118 vs 0.260 | +0.859 [+0.710, +1.023] | +6 165 [+5 577, +6 757] |
| US-101 | gru_wmax - gru | 1.118 vs 0.421 | +0.698 [+0.566, +0.846] | +6 252 [+5 680, +6 850] |
| US-101 | lstm_wmax - lstm_windowed | 0.926 vs 0.679 | +0.247 [+0.142, +0.370] | +1 272 [+821, +1 759] |
| US-101 | lstm_wmax - lstm | 0.926 vs 0.888 | +0.038 [-0.064, +0.142] | +1 251 [+571, +1 916] |
| US-101 | perl_wmax - perl_windowed | 0.789 vs 0.904 | -0.115 [-0.205, -0.040] | -154 [-829, +578] |
| US-101 | perl_wmax - perl | 0.789 vs 0.661 | +0.128 [+0.047, +0.235] | -1 286 [-1 890, -605] |

**Secondary numbers of D134 in the corridor** (no verdict): gru_wmax against gru_windowed (M11: 0.132 / 0.260, contacts
270-298) has the macro error 0.597 / 1.118 (+0.47 and +0.86, both intervals above zero) and 6 384 / 6 463 contacts per
1 000 vehicle-km, with 79 % / 61 % of the planned vehicles inserted (gru_windowed: all); lstm_wmax against
lstm_windowed (0.423 / 0.679) 0.481 / 0.926 (+0.06 [-0.02, +0.13] on I-80, +0.25 [+0.14, +0.37] on US-101) and more
contacts (3 386 / 4 108 against 2 666 / 2 836). The members of the laws of D134 are the most stable of their
architectures on the NGSIM band (unstable among the equilibria 0.12, 0.23, 0.15 against 0.37, 0.09, 0.83 with the mean
and 0.72, 0.73, 0.94 free), yet GRU and LSTM are worse corridor laws than with the mean, and gru_wmax is the worst
GRU law of the study (on US-101 the largest macro error of every law in the tables); PERL with the maximum is better than with the mean on both corridors (-0.11 and -0.12) and still collides
as much as free PERL.

### A'.9 Deviations and observations for the decision log

1. The info `argmax_omega` is recorded under both reductions (the contract names it for the history of the new runs):
   every new run of `windowed_gain` has the two history keys `penalty_argmax_omega` and `val_penalty_argmax_omega` more;
   the M11 runs keep their histories (nothing of M11 is trained again: the hashes are unchanged, A'.2). The value of
   the training steps is the mean over the penalty steps of the epoch (the trainer averages every info entry), that of
   the validation one audit frequency.
2. D85 on the pilot: GRU 10 and 100 both reach 1.00 (the smaller validation RMSE: 100), LSTM 1 and 10 reach 1.00 and
   0.95 (the smaller validation RMSE: 10), PERL reaches 0.9 at no weight (the largest share, 0.05 at 100; no tie). The
   rule takes the validation RMSE among the weights that reach 0.9; GRU at weight 1 (+4.5 % on fold 0, stable 0.82) is
   below the share and was not chosen.
3. The windowed linearisation and the numerical audit disagree more for the GRU with the maximum than for the mean:
   pilot weight 1 linearly stable at every speed but amplifying a 0.2 m/s oscillation by up to 1.15 at 1.36-1.65 rad/s
   at 8-11 m/s (fit residuals 0.08-0.11, a non-sinusoidal response); chosen weight 100 with linear gains above 1.02 at
   0.5-2 rad/s at 20 % of its speeds that the rollouts do not find (agreement 0.80, A'.6). The rule of H-N3' uses the
   numerical audit, as D128.
4. `scripts/analysis/windowed_gain.py` was refactored around an `Arm` (a named tuple: the tests load the script without
   registering a module, which a dataclass needs) instead of copied; the M11 tables it writes are byte-identical to git
   (A'.2).
5. The new statistic of the arm set m12 was first named `windowed_unstable_high`, which collides with the upper interval
   bound of `windowed_unstable` (`<statistic>_high`): the table of 23:08 showed wrong upper bounds in the column
   "windowed > 1.02". Renamed `windowed_unstable_highfreq` (a test now checks every statistic against its interval and
   the absence of such collisions) and the table rewritten at 23:10, before any use.
6. The long-window check and the full-history audit of the arm set m12 also include the M11 pilot weights (fold 0)
   beside the M12 pilot, for the comparison of the same weights under both reductions; their results equal those of the
   M11 tables and nothing is written into their runs.
7. Line endings: several shared files that the index holds with LF are CRLF in the working tree after the edits of
   agent B' (`cf_stability/eval/corridor_tables.py`, `cf_stability/eval/figures_supplement.py`,
   `configs/corridor_metrics.yaml`, among others); `git diff` then shows whole files, `git diff --ignore-all-space` the
   changes. My own new files and the LF files I edited alone are LF; the CRLF files I edited (`penalties.py`,
   `train.yaml`, `tests/test_penalties_windowed.py`, `tests/test_run_experiment.py`) kept CRLF.

### A'.10 Commands run, run counts and durations

```bash
for q in configs/queue/*.yaml; do python scripts/run_experiment.py $q --dry-run; done      # before and after the key
python scripts/make_tables.py                                                             # before the change: m4 unchanged
python scripts/analysis/windowed_gain.py tables                                           # before the refactoring: m11 unchanged
python -m pytest tests/test_penalties_windowed.py tests/test_utils.py tests/test_penalties.py
CF_GPU_TESTS=1 python -m pytest tests/test_penalties_windowed.py -k "cuda or graph or eager"
python scripts/run_experiment.py configs/queue/e2_wmax_pilot.yaml                         # 12 runs, 18:20-21:14
python scripts/make_tables.py                                                             # after the change: m4 unchanged
python scripts/analysis/full_history_audit.py --no-json --no-figure                       # m9 table unchanged
python scripts/analysis/full_history_audit.py --arm-set m11 --no-json                     # m11 table unchanged
python scripts/audit_stability.py run='<scratchpad copy of e2_windowed_w10/.../gru/driver_fold0_seed0>'   # audit path check
python scripts/analysis/full_history_audit.py --arm-set m12                               # 21:14, the pilot runs (51 runs)
python scripts/analysis/windowed_gain.py choose --arm wmax                                # D85: gru 100, lstm 10, perl 100
python scripts/run_experiment.py configs/queue/e2_wmax.yaml                               # 12 runs, 21:16-23:08
python scripts/analysis/long_window_check.py --arm-set m12 --device cuda                  # 24 runs, 21:19-21:33
# chained (scratchpad m12_A/chain.sh, nohup, log m12_A_chain.log) from the end of e2_wmax:
python scripts/run_experiment.py configs/queue/m12_ft_wmax.yaml                           # 15 runs, from 23:08
python scripts/analysis/full_history_audit.py --arm-set m12                               # 66 runs, 23:08 (again 23:10)
python scripts/analysis/windowed_gain.py tables --arm wmax                                # 23:09: pilot, five folds, branch
python -c "export_laws(yaml.safe_load(open('configs/corridor/laws.yaml')), 'runs/corridor/laws', only=['gru_wmax', 'lstm_wmax', 'perl_wmax'])"
python scripts/check_environment.py
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] \
    laws=[gru_wmax,lstm_wmax,perl_wmax] seeds=[0,1,2,3,4,5,6,7,8,9] workers=3          # 180 runs (and a second pass)
python scripts/corridor_metrics.py law=gru_wmax   # and lstm_wmax, perl_wmax
python scripts/corridor_asymmetry.py law=gru_wmax workers=2   # and the other two
python scripts/make_tables.py
python scripts/corridor_metrics.py compute=false tables=true
python scripts/analysis/wmax_pairs.py
python scripts/analysis/windowed_gain.py tables --arm wmax                                # with the fine-tuned rows
python -m pytest tests/test_penalties_windowed.py tests/test_penalties.py tests/test_utils.py tests/test_windowed_analysis.py \
    tests/test_corridor_tables_wmax.py tests/test_corridor_tables_windowed.py tests/test_run_experiment.py tests/test_trainer.py
```

The corridor chain ran 00:39-02:52 (law export 00:39, environment check 00:39, runs 00:39-02:48, metrics and asymmetry
02:48-02:50, tables 02:50-02:52). Logs: scratchpad `m12_A_*.log` (`m12_A_chain.log` lists every step with its time and
exit status).

| step | runs | wall time | per job |
|---|---:|---|---|
| e2_wmax_pilot (train, audit, platoon) | 12 | 2026-10-20 18:20-21:14 | 10.1-48.4 min (training 8.4-44.6 min, 13-94 epochs) |
| long-window check, arm set m12 (GPU) | 24 (12 written) | 21:19-21:33 | |
| e2_wmax (train, audit, platoon) | 12 new + 3 pilot | 21:16-23:08 | 9.4-32.7 min |
| full-history audit, arm set m12 (CPU) | 66 | 23:08, 23:10 | 12 s in all |
| m12_ft_wmax (train, audit) | 15 | 23:08-00:39 | 4.6-19.4 min |
| corridor runs | 180 | 2026-10-21 00:39-02:48 (workers 3) | 106-159 s |
| macro.json and asymmetry.json | 180 + 180 | 02:48-02:50 | |
| tables (make_tables, corridor tables, wmax_pairs, windowed_gain) | | 02:50-02:52 | |

No job failed and nothing was rerun. The queues waited for free memory (below 6 GB) at times during the pilot.

## B': two-timescale certified law with a persistent latent (D135, H-N2')

Status: complete (20 October, 23:50; started 17:53). Done: implementation, tests (B'.7), pilot and choice of tau_z
(B'.3), the five folds on highD (B'.4), the fine-tuning, the frozen and online audits and the 21-z certificate
(B'.5), the law residual_idm_tt2 with its 60 corridor runs, metrics, asymmetry, the corridor tables and the
paired table tt2_pairs (B'.5b).

### B'.0 Baseline before any change

- Dry runs (`python scripts/run_experiment.py configs/queue/<file>.yaml --dry-run`, 2026-10-20 18:00-18:05, before any
  edit of agent B'): e1 175 jobs complete, e4 100 complete, e4_rmax 200 complete, m8_temporal 15 complete, e2_windowed
  15 complete, e4_variant_d 50 (10 complete, 40 with a later step to run), e4_tt 10 and e4_tt_pilot 3 (train complete,
  the step `certificate` listed because the files of these runs were all written at 16:11:40 by the checkout and
  `certificate.json` is a few microseconds older than `model.pt`); output kept in the scratchpad (`m12_B/dry_before/`).
- Event lengths (fold 0, driver split): follownet_highd 150 samples for every event (7 506 / 2 503 / 2 503 events);
  ngsim_i80 150-2 171 samples (median 383, mean 485; 4 584 / 1 566 / 1 501 events).

### B'.1 What was done (code)

- `cf_stability/models/residual_idm_tt2.py` (new), registered as `residual_idm_tt2` (`cf_stability/models/__init__.py`),
  model config `configs/model/residual_idm_tt2.yaml` (as `residual_idm_tt`, plus `z_lo: -0.2`, `z_hi: 1.0`).
  `ResidualIDMTT2` subclasses `ResidualIDMTT` (the file of D129 is not touched): the target of the latent
  `z_target = z_lo + (z_hi - z_lo) sigmoid(g_phi(h_t))`, `T(z) = T_c exp(z ln 1.5)`; the bias of the head starts at
  `ln(-z_lo / z_hi)` = ln 0.2, so that the untrained target is 0 (1.5e-8 in float32) and the untrained model is the
  margin core, as in D129; `LATENT_SWEEP` = the 21 values -0.2, -0.14, ..., 0.94, 1.0. Persistence: `stateful` (True,
  False with a frozen latent), `initial_state(batch, device)` (z = 0), `forward(history, state=None)`: without a state
  the model of D129 with the new target (the window-restarted latent, the code path of `ResidualIDMTT.forward`); with a
  state `(acceleration, z_next)`, `z_next = z + (dt / tau_z)(z_target(window's last step) - z)`, the acceleration with
  `z_next` (as the window latent of D129 includes the last state). `steady_state(history)` = the target of the window
  (the fixed point for a constant history); `window_drive`, `carry` (closed form of k steps) and `carry_path` (the
  latent after every step) for the batched teacher-forced passes; `latent_range()` = (z_lo, z_hi).
- `cf_stability/models/base.py` (additive): `CFModel.stateful = False`, `initial_state` and `steady_state` (None for a
  stateless model), `step(history, state)`; `SteadyStateView` / `steady_view(model)`: the stateful model with its
  state at the steady state of the history, the memoryless law whose zeros are its equilibria.
- Callers that carry the state:
  - trainer (`cf_stability/train/trainer.py`, new `cf_stability/train/stateful.py`): before the rollouts of a step the
    teacher-forced pass over every event of the batch (once per event, up to the latest start of its rollouts, shared by
    its rollouts), gradients kept: the host plans the windows (`pass_plan`), the device computes the targets of all
    windows in one batch and the state at every start in closed form (`states_at_starts`); the plan has a fixed
    capacity (`pass_capacity`: 1.25 x the largest pass of 256 random batches, at most 256 x (T_max - 29)), so that the
    step keeps fixed shapes and is replayed from a CUDA graph; a batch beyond the capacity runs eagerly with a larger
    plan (`training.state_pass.overflow_steps`). The rollouts carry the state (`rollout_loss(..., state=)`,
    `rollout_model(..., state=)`); the validation and test rollouts start from z = 0 at the first window of every event
    (`evaluate_closed_loop`), the open-loop error uses the teacher-forced state (`teacher_forced_path`). The acceleration
    loss of a training step (weight 0 for this model, D63) still calls the model without state.
  - audit (`cf_stability/stability/audit.py`, `frequency.py`): the equilibria and partials of the steady-state view,
    the numerical frequency response from the steady state of the warm-up on the equilibrium, carried; the analytic gain
    "discrete" (the windowed linearisation does not hold the latent); `state: "steady"` in the audit. The audits at a
    frozen latent are unchanged (a frozen model is stateless).
  - `rollout_gain` of `cf_stability/stability/penalties.py` (file of agent A'; only this function touched, an optional
    argument `state` appended): a stateful model starts from its steady state on the equilibrium unless a state is
    given.
  - platoon test (`cf_stability/stability/platoon.py`): the start gaps from the steady-state view, one state per
    follower from the steady state of its start history, carried.
  - corridor (`cf_stability/corridor/laws.py`): a kind `models` law with stateful members gets `stateful: true` in its
    law file (a spec may set `stateful: false`), `ModelLaw` keeps one state per vehicle (the initial state of its member
    until its first step under the law, then carried); the key is written only for stateful members, so the files and
    fingerprints of every other law are unchanged.
- Certificate (`cf_stability/stability/certificate.py`, `scripts/certificate.py`): `latent_sweep(model)` (the model's
  `LATENT_SWEEP`, else the 21 values of [-1, 1]), `latent_interval(model)` (the range, else [-reach, reach]),
  `reachable_latent` (a model with a range: its sweep), `enforce_budget` records `latent.range`; `certify_latent` sweeps
  the model's own 21 values; the records of the models of D129 are unchanged. `scripts/train.py`: `residual_idm_tt2` in
  `NEEDS_IDM` and `CERTIFIED`.
- Analysis scripts: `scripts/analysis/tt2_summary.py` (new, the sibling of `tt_summary.py`, whose helpers it imports:
  tables `m12/tt2_pilot`, `tt2_highd`, `tt2_ngsim`, the persistent latent on the test events, the choice of tau_z and
  the branch of H-N2'), `scripts/analysis/m12_pairs.py` (new, as `m11_pairs.py`: table `m12/tt2_pairs`),
  `scripts/analysis/certificate_tt.py` (option `--arm tt2`: table `m12/certificate_tt2`; the default arm unchanged).
- Queue files `configs/queue/e4_tt2_pilot.yaml` (3 runs) and `configs/queue/e4_tt2.yaml` (e4_tt2 and its fine-tuning
  e4_tt2_ft in one file, as in M11).
- Appended blocks of shared files: `configs/corridor/laws.yaml` (the block "M12 (agent B')": law `residual_idm_tt2`,
  kind models, cuda, the members `runs/e4_tt2_ft/ngsim_i80/residual_idm_tt2/driver_fold{fold}_seed0`),
  `configs/corridor_metrics.yaml` (`design.m12_laws: [residual_idm_tt2]`), `cf_stability/eval/corridor_tables.py`
  (`m12_laws` handled as `m11_laws`: optional, rows of laws, components, instability, asymmetry and contacts_absolute,
  `m12_note`; not in the correlations nor the verdicts), `cf_stability/eval/figures_supplement.py` (`m12_laws` in
  `EXCLUDED_LAW_SETS`).

### B'.2 Cost of the teacher-forced pass (measured before the runs)

Training step of fold 0 (`scratchpad/m12_B/bench_step.py`, 40 resp. 30 steps, shared GPU under load): highD
`residual_idm_tt` 87 ms with the CUDA graph (293 ms eager), `residual_idm_tt2` 137 ms (259 ms eager), pass capacity
24 576 windows, 1.5 GB; NGSIM I-80 115 ms (332 eager) and 489 ms (480 eager), capacity 126 976 windows (n_max 2 142),
6.9 GB. On I-80 the pass dominates: the cuDNN GRU over the ~84 000 windows of a batch takes about 180 ms forward and
backward (a hand-written GRU was slower, 558 ms); the graph and the eager step give the same weights bit for bit (test
B'.7). Consequence: highD runs of about 20 min, I-80 fine-tuning runs of about an hour.

### B'.3 Pilot (`e4_tt2_pilot_tau{15,30,60}`, highD fold 0, seed 0; table `m12/tt2_pilot`)

| tau_z | val RMSE s (m) | test RMSE s (m) | epochs (best) | admissible budget | unstable eq. at z = -1, -0.5, 0, 0.5, 1 (and -0.2) | online: unstable eq., max gain | band stable at z = -1, 0, 1 | certificate: z of 21 | latent on the test events: 1 % / median / 99 %, max \|z\| | share \|z\| > 0.182 | bound in a 15-s event |
|---:|---:|---:|---|---:|---|---|---|---:|---|---:|---:|
| 15 s | 4.368 | 4.266 | 59 (49) | 0.0765 | 0 at all (0) | 0, 1.000 | 1.000, 0.955, 0.864 | 21 | -0.111 / -0.001 / 0.555, 0.555 | 0.303 | 0.555 |
| 30 s | 4.478 | 4.382 | 12 (2) | 0.0765 | 0 at all (0) | 0, 1.002 | 1.000, 0.955, 0.864 | 21 | -0.065 / -0.001 / 0.320, 0.323 | 0.151 | 0.332 |
| 60 s | 4.530 | 4.435 | 12 (2) | 0.0765 | 0 at all (0) | 0, 1.000 | 1.000, 0.955, 0.864 | 21 | -0.036 / -0.000 / 0.176, 0.178 | 0.000 | 0.183 |

M11 (D129) on the same fold: val / test 4.271 / 4.170 (15 s), 4.421 / 4.321 (30 s), 4.488 / 4.392 (60 s). Budget: the
smallest admissible budget over the 21 latents of [-0.2, 1] is that of z = -0.2 (0.0765, lipschitz 0.2295; M11 0.0802 at
|z| <= 0.182). Every run: 26 equilibria at every frozen z and online, the band shares below 1 only through equilibria
outside the band at larger T (none unstable); the scanned a priori certificate holds at all 21 z (min 0.005 s^-2);
platoon growth error 0.178 / 0.172 / 0.170, no collision; no eager pass step (capacity 24 576 windows, n_max 121).

**Rule of D129 with the online audit of H-N2'**: all three runs have no unstable equilibrium at any frozen z of D129 and
in the online audit, so tau_z = the smallest validation RMSE = **15 s** (configs/queue/e4_tt2.yaml).

**The latent's range on the test events** (the teacher-forced persistent latent from z = 0 before the first window,
every 10th window end; 32 539 windows): it now spreads beyond the window bound 0.182 of D129 (99 % quantile 0.555 at
15 s, 30 % of the windows beyond 0.182), but it is bounded by the length of a highD event: within 121 windows from z = 0
it cannot pass (1 - (1 - dt/tau_z)^121) x 1 = 0.555 / 0.332 / 0.183 at 15 / 30 / 60 s, and the 99 % quantile sits at
that bound (the targets of these events are at z_hi = 1). The median stays at 0 (T = T_c), the 1 % quantile at -0.11:
the latent separates a group of slow (large-gap) followers rather than spreading evenly. At tau_z 60 s the latent cannot
leave the old bound within a highD event at all.

### B'.4 Five folds on highD (`e4_tt2`, seed 0, tau_z 15 s; table `m12/tt2_highd`)

| fold | test RMSE s (m) | GRU E1 (m) | ratio to GRU | E4 (m) | ratio to E4 | tt of D129 (m) | ratio to tt | epochs | admissible budget | unstable eq. at the 5 frozen z (and -0.2) | online: unstable eq., max gain | certificate: z of 21 | latent 1 % / median / 99 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| 0 | 4.266 | 2.063 | 2.067 | 4.470 | 0.954 | 4.170 | 1.023 | 59 | 0.0765 | 0 at all (0) | 0, 1.000 | 21 | -0.111 / -0.001 / 0.555 |
| 1 | 4.394 | 2.139 | 2.055 | 4.581 | 0.959 | 4.295 | 1.023 | 28 | 0.0777 | 0 at all (0) | 0, 1.001 | 21 | -0.111 / -0.001 / 0.555 |
| 2 | 4.297 | 2.086 | 2.060 | 4.481 | 0.959 | 4.191 | 1.025 | 28 | 0.0779 | 0 at all (0) | 0, 1.000 | 21 | -0.111 / -0.001 / 0.555 |
| 3 | 4.414 | 2.197 | 2.009 | 4.603 | 0.959 | 4.304 | 1.026 | 24 | 0.0789 | 0 at all (0) | 0, 1.000 | 21 | -0.111 / -0.001 / 0.555 |
| 4 | 4.240 | 1.952 | 2.172 | 4.425 | 0.958 | 4.131 | 1.026 | 32 | 0.0779 | 0 at all (0) | 0, 1.002 | 21 | -0.111 / -0.001 / 0.555 |
| mean | 4.322 | 2.087 | **2.071** | 4.512 | 0.958 | 4.218 | 1.025 | 34.2 | 0.0778 | | | | |

The ratios of the mean row are the ratios of the 5-fold means. Fold 0 is the pilot run of tau_z 15 s trained again
under its own experiment name: the same validation and test RMSE to the last digit (4.368401, 4.265698). Every fold: 26
equilibria at every frozen z and online, band share stable 1.000 at z = -1, 0.955 at -0.5, -0.2 and 0 and online,
0.909 at 0.5, 0.864 at 1 (the rest: equilibria outside the band at larger T, none unstable); largest gain at z = -1
1.0016-1.0019; platoon growth error 0.175-0.178, no collision; the scanned a priori certificate holds at all 21 z of
[-0.2, 1] on every fold before fine-tuning. Training 24-59 epochs (13.8-44.0 min per job of the queue); no eager pass
step (capacity 24 576-28 672 windows).

Secondary numbers of D135 (highD): against E4 the persistent law is 4.2 % more accurate (ratio 0.958; M11: 0.935),
against the law of D129 with the window-restarted latent 2.5 % less accurate (ratio 1.025, every fold 1.023-1.026).
The latent's spread on the test events is the same on every fold: 1 % / median / 99 % = -0.111 / -0.001 / 0.555, max
|z| 0.555 (the bound of a 15-s event), 28.5-30.3 % of the windows beyond 0.182 (M11: 1 % / 99 % at -0.182 / +0.182 on
every fold, the window bound).

### B'.5 Fine-tuning on NGSIM I-80 (`e4_tt2_ft`; tables `m12/tt2_ngsim`, `m12/certificate_tt2`) and the rule of H-N2'

`init_from` the e4_tt2 run of the same fold (the checkpoint as it is: core, scalers, certified budget), the latent
carried along the I-80 events (15-217 s) in the training pass and the evaluation:

| fold | test RMSE s (m) | E4 fine-tuned (m) | ratio to E4 | tt fine-tuned (m) | ratio to tt | epochs | unstable eq. at the 5 frozen z (and -0.2) | online: unstable eq., max gain | certificate: z of 21 (scan) | min guaranteed margin (s^-2) | latent on the I-80 test events: 1 % / median / 99 %, max \|z\| |
|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| 0 | 4.687 | 4.916 | 0.953 | 4.810 | 0.974 | 15 | 0 at all (0) | 0, 1.000 | 21 | 0.0053 | -0.197 / 0.129 / 0.966, 0.9998 |
| 1 | 4.723 | 4.931 | 0.958 | 4.818 | 0.980 | 11 | 0 at all (0) | 0, 1.000 | 21 | 0.0054 | -0.193 / 0.150 / 0.966, 0.9989 |
| 2 | 4.706 | 4.892 | 0.962 | 4.797 | 0.981 | 26 | 0 at all (0) | 0, 1.000 | 21 | 0.0054 | -0.197 / 0.111 / 0.949, 0.9987 |
| 3 | 4.654 | 4.849 | 0.960 | 4.760 | 0.978 | 13 | 0 at all (0) | 0, 1.000 | 21 | 0.0050 | -0.195 / 0.100 / 0.959, 0.9994 |
| 4 | 4.656 | 4.845 | 0.961 | 4.746 | 0.981 | 19 | 0 at all (0) | 0, 1.000 | 21 | 0.0055 | -0.196 / 0.130 / 0.956, 0.9995 |
| mean | 4.685 | 4.887 | 0.959 | 4.786 | 0.979 | 16.8 | | | | | -0.196 / 0.124 / 0.959 |

After fine-tuning every fold has 26 equilibria at every frozen z and online, band share stable 1.000 at every frozen z
and online, largest gain at z = -1 1.0028-1.0032, online at most 1.0000; platoon growth error 0.173-0.189, no
collision; no eager pass step (capacity on I-80 about 127 000 windows). Jobs of 16.3-43.6 min (11-26 epochs at about
77 s per epoch, early stopping). On the I-80 events the persistent latent uses its whole range: 1 % / median / 99 % =
-0.196 / 0.124 / 0.959 over every 10th window end (max |z| 0.9998), 48-51 % of the windows beyond 0.182, at the event
ends median 0.33 and 99 % 0.98 (a 217-s event allows 1 - (1 - dt/tau_z)^2 096 = 1.0 of the target).

**Exact certificate at the 21 latents of [-0.2, 1]** (`python scripts/analysis/certificate_tt.py --arm tt2`, table
`m12/certificate_tt2`; the machinery of `certificate_exact.py` at the grid speeds and on the sweep of 2 501 speeds):
before and after fine-tuning it holds for all 5 runs at all 21 z (exact grid, sweep and scan). Smallest exact margin
0.0054 s^-2 before and 0.0050 after fine-tuning, both at z = -0.2 (T = 0.922 T_c); 0.028 at z = -0.02, 0.064 at z = 1;
the largest at z = 0.46 (0.077); exact, sweep and scan agree (largest |exact - scan| 1.7e-8, the stored scans
reproduced from the stored inputs); min a(1 - (v/v0)^4) - r_max = 0.439 m/s^2 (every residual within r_max leaves an
equilibrium in I(v)). M11 (D129) held at 13 of its 21 z of [-1, 1] (none below -0.2).

**Rule of H-N2'** (D135; 5-fold means on highD): test spacing RMSE 4.322 m = **2.071 x** the GRU's E1 RMSE (2.087 m),
above 1.15 x: the numbers fall in the branch **refuted**. Both stability conditions of the branch "confirmed" are met:
every frozen-z audit (the five z of D129, every fold, before and after fine-tuning) and the online audit (the
persistent latent) have no unstable equilibrium, and the certificate holds at all 21 z of [-0.2, 1] after fine-tuning
(scanned and exact, every fold). (M11, H-N2: 2.02 x; certificate at 13 of 21 z.)

### B'.5b Corridor (law `residual_idm_tt2`; tables `m12/tt2_pairs`, rows of `runs/_tables/m5` and `m8`)

Law `residual_idm_tt2` (`runs/corridor/laws/residual_idm_tt2.json`, fingerprint `ae3d76f5c2f8`, `stateful: true`; the
five fine-tuned folds, drawn per vehicle as `residual_idm_certified` draws them, inference on the GPU, one latent per
vehicle from z = 0 at its insertion), exported alone (`export_laws(..., only=[residual_idm_tt2])`: no other law file
rewritten). 60 runs (i80_p0-p2, us101_p0-p2, seeds 0-9; 22:06-23:36, 3 workers), all complete: every planned vehicle
inserted, 20 contact episodes (all on US-101), one teleport (i80_p1). Paired by scenario and seed against E and against
the law of D129 (`scripts/analysis/m12_pairs.py`, the builder of `ablation_pairs`):

| corridor | tt2 | E (residual_idm_certified) | tt (D129) | tt2 - E [95 % interval], p | tt2 - tt [95 % interval], p | dynamic macro error tt2 / E / tt | rank of tt2 by macro error |
|---|---:|---:|---:|---|---|---|---|
| I-80 | 0.1272 [0.1107, 0.1447] | 0.1248 | 0.1178 | +0.0024 [-0.0261, +0.0266] (no difference), 0.95 | +0.0094 [-0.0111, +0.0311] (no difference), 0.78 | 0.224 / 0.190 / 0.182 | 5 of 32 laws |
| US-101 | 0.0737 [0.0692, 0.0784] | 0.1129 | 0.0902 | -0.0392 [-0.0473, -0.0311] (lower; -35 % [-40, -29]), < 0.0001 | -0.0165 [-0.0241, -0.0099] (lower; -18 % [-26, -11]), 0.0001 | 0.087 / 0.122 / 0.082 | 1 of 27 laws |

Dynamic macro error: on US-101 lower than E (-0.035 [-0.051, -0.021]) and no different from tt (+0.005 [-0.009,
+0.016]); on I-80 no difference from either (+0.034 [-0.024, +0.084] against E). Components (mean over the runs): on
US-101 the errors of the mean speed (0.072 against E 0.134 and tt 0.125), the queue discharge flow (0.074 against 0.162
and 0.163), the fundamental diagram (0.076 against 0.096 and 0.082) and the travel time (0.078 against 0.124 and 0.118)
are smaller; the wave speed error is that of E (-0.085 against -0.092; tt -0.020). On I-80 the mean speed (0.026 against
0.050 / 0.041), the queue discharge flow (-0.030 against -0.077 / -0.056), the wave amplitude (-0.072 against +0.162 /
+0.092) and the travel time (0.062 against 0.080 / 0.075) are closer to the truth, the wave speed (-0.081 against -0.023 /
-0.022) and the number of waves (-0.318 against -0.293 / -0.298) further. Contacts per 1000 vehicle-km: I-80 0 (E 0, tt
0); US-101 0.489 (13 of 30 runs with a contact, 0.50 episodes per run) against E 0.227 (higher: +0.262 [+0.016, +0.536],
p 0.039) and tt 0.245 (+0.244 [-0.054, +0.545], no difference). Instability of the members (their audits: the
persistent latent at its steady state, the rollouts carrying it): no unstable equilibrium, band share not stable 0, on
both corridors. Acceleration asymmetry index: I-80 1.287 (E 1.281, tt 1.247), US-101 1.414 (E 1.488, tt 1.409).

**Secondary numbers of D135** (reported with intervals, no verdict): RMSE against E4 0.958 x (highD, 5-fold means;
M11: 0.935) and 0.959 x after fine-tuning on I-80; against the law of D129 1.025 x on highD and 0.979 x on I-80; the
latent's spread: highD test events 1 % / median / 99 % = -0.111 / -0.001 / 0.555 (the bound of a 15-s event), I-80 test
events -0.196 / 0.124 / 0.959 (M11: the window bound +-0.182 on every fold); the corridor macro error against E 0.127 /
0.074 (E 0.125 / 0.113) and against tt (0.118 / 0.090), I-80 / US-101.

### B'.6 Deviations and choices left open by the contract

1. **`state=None` and "the M11 model bit for bit".** Without a state the tt2 model runs the code path of D129 (the latent
   integrated over the window from z = 0) with its own target in [-0.2, 1]; the M11 model (target tanh) is reproduced bit
   for bit when the stored e4_tt checkpoint is put into the persistent class with the target of D129 (test of B'.7). The
   two cannot coincide with the new target itself, which the contract changes.
2. **Initial target.** The bias of the head starts at ln(-z_lo / z_hi) = ln 0.2 (weights zero), so that the untrained
   target is 0 and the untrained model is the margin core, as the zero head of D129 (float32: 1.5e-8).
3. **What is carried.** Only the latent z is the state; the GRU of the latent still reads the window (its hidden state
   restarts with every window: the fast part is a feature of the window, the slow latent the state of the vehicle). A
   call advances z once with the target of the window's last step and the acceleration uses the advanced value (as the
   latent of D129 includes the last state of its window).
4. **Trainer.** The event-level pass runs within the existing loader (`EventTensors` holds the whole events), so the
   fallback of D136 was not needed. The pass is planned on the host with a fixed capacity so that the step stays in its
   CUDA graph; a batch beyond the capacity runs eagerly (none in the pilot). The acceleration loss of the step (weight 0
   for this model, D63) still calls the model without a state; the open-loop error of `metrics.json` uses the
   teacher-forced state.
5. **"The state initialised at the warm-up on the equilibrium"** read as the steady state of the warm-up history (the
   latent at the target of the constant window, `steady_state`), in the audit, `rollout_gain` and the platoon test: z = 0
   advanced over the 30 warm-up samples would leave the window value of D129 and let the latent drift through the
   measurement (with tau_z 60 s by a third of its distance over the 60 s discarded). The equilibria of a stateful model
   are those of its steady state (`SteadyStateView`), consistently.
6. **Analytic part of the audit of a stateful model**: the partials and the analytic criterion are those of the
   steady-state view (the quasi-static limit w -> 0, the latent following its target), the analytic gain is the discrete
   one of these partials; the windowed linearisation of the window does not hold the latent's dynamics, so the agreement
   statistics of the audit are approximate for this model. The numerical frequency response, which decides every
   stability statement here, carries the latent.
7. **Corridor.** A vehicle's state is the initial state (z = 0) until its first step under the law, its insertion into
   the controlled section, where the loop fills its history with its first state; then it is carried. The law file gets
   `stateful: true` (a spec may set `stateful: false`, the window-restarted latent of D129), which enters the
   fingerprint and hence the hashed config of the runs; `ModelLaw` without the key is unchanged.
8. **Frozen-z audits**: the five z of D129 (-1, -0.5, 0, 0.5, 1; the rule) plus z = -0.2, the lower end of the range,
   for information. z = -1 and -0.5 lie outside the range of the latent and of its certified budget.
9. **Choice of tau_z**: the rule of D129 (no unstable equilibrium at any frozen z, then the smallest validation RMSE)
   with the online audit of H-N2' added to the eligibility; all three runs were eligible, so the choice is the same either
   way.
10. **Ownership notes.** `rollout_gain` of `penalties.py` (agent A'): only this function, one appended argument and the
    lines that use it. `cf_stability/eval/corridor_tables.py`: the block `m12_laws` and the field `filter_laws` (the
    blocking error of the committed state, B'.8). `tests/test_models.py`, `tests/test_run_experiment.py`: appended
    entries. The report section was written again from a copy in the scratchpad after a rewrite of the file by another
    agent (about 18:25) had removed it.
11. **Stray files.** A first attempt of the frozen-z audits passed Git Bash paths (`/c/Users/...`) to Python, which read
    them as `C:\c\Users\...`: six error files `stability_z*.json` were written under
    `C:\c\Users\Город\Downloads\research_plans\cf-stability\runs\e4_tt2_pilot_tau30\follownet_highd\residual_idm_tt2\driver_fold0_seed0\`
    (outside the repository); the removal of `C:\c` was refused by the safety check of the session, so the directory is
    left for the user to delete. The audits were then run with relative paths.

### B'.7 Tests

- `tests/test_residual_idm_tt2.py` (17, one GPU test opt-in with `CF_GPU_TESTS=1`): the untrained model is the margin
  core (target 0 to 1e-7), the range, the sweep and the frozen model; the target stays in [-0.2, 1]; **`state=None` is the
  M11 model bit for bit on the stored e4_tt run** (fold 0: the checkpoint of D129 loaded into the persistent class with the
  target of D129, single calls and 200-step closed-loop rollouts `torch.equal` in float32 and float64; the persistent
  class with its own target is the window latent of D129 in closed form); **the persistent latent of a constant input
  converges to its target with the time constant tau_z** (z_k = z* (1 - (1 - dt/tau_z)^k) to 1e-12, 1 - 1/e after
  tau_z, the target after 10 tau_z, while the window latent stays below 0.182) and stays in [-0.2, 1] on random inputs;
  `carry` and `carry_path` equal the recursion; gradients reach the residual, the GRU and the head through the carried
  latent; the steady-state view and its equilibria; checkpoint and factory; **the trainer's pass on hand-made events gives
  at every rollout start the latent of the sequential calls from the first window** (several starts in one event, one at
  the first window, float64 to 1e-12, float32 to 2e-5), the pass plan (shared events, padding behind the rows, capacity),
  the pass contributes to the gradient (detaching it changes the gradient of the head), a short training on the CPU and
  the closed-loop evaluation from z = 0 at the first window (equal to the explicit rollout; without the state the
  rollout differs), the open-loop error; a stateless model gets no `state_pass`; on the GPU the steps replayed from the
  CUDA graph give the weights and history of the eager steps bit for bit (`CF_GPU_TESTS=1`: passed).
- `tests/test_stateful_callers.py` (6): a toy stateful model that counts its calls in its state, starts at 1000 from its
  steady state and at 0 from its initial state, adds `1e-6 (state - 1000)` to the IDM and records the state of every
  call: the audit's frequency-response rollouts (`excite`, and `audit_model` as a whole: equilibria of the steady-state
  view, `state: "steady"`, the longest rollout reaches 1000 + its number of calls - 1), `rollout_gain` (from the steady
  state, or from a given state), the platoon test (every follower of every profile, one call per step, from the steady
  state), the closed-loop evaluation (from 0 at the first window), the trainer (the rollouts start from the sequential
  pass, `j - window + 1` calls), `ModelLaw` (one state per vehicle, a later vehicle from 0, members apart; without
  `stateful` no state at all).
- `tests/test_corridor_laws_m12.py` (3; the loop test skipped without the eclipse-sumo package): the law file of tt2
  members has `stateful: true` (`false` from the spec; no key for stateless members), the fingerprint differs between the
  two; `ModelLaw` reproduces the sequential latent of two vehicles entering at different steps; **the corridor loop on the
  synthetic scenario of `tests/test_corridor_loop.py`** with the same tt2 members with and without persistence: different
  trajectories, the same members, every vehicle inserted, the run config hash holds the law fingerprint (different for the
  two).
- Shared tests (appended): `tests/test_models.py` (`residual_idm_tt2` in the tables of the registry) and
  `tests/test_run_experiment.py` (the queue files `e4_tt2_pilot`, 3 jobs, and `e4_tt2`, 10): both pass (the queue-file
  test with the entries of agent A' added as well, 23:55).
- Existing tests of the touched modules rerun: `tests/test_closed_loop.py`, `test_evaluate.py`, `test_trainer.py`,
  `test_residual_idm_tt.py`, `test_certificate_latent.py`, `test_frequency.py`, `test_platoon.py`, `test_audit.py`,
  `test_train_script.py`, `test_platoon_script.py`, `test_certificate.py`, `test_certificate_script.py`,
  `test_certificate_exact.py`, `test_corridor_laws_m8.py`, `test_corridor_laws_review.py`, `test_residual_idm.py`,
  `test_tensors.py`, `test_corridor_tables*.py`, `test_figures_supplement.py`, `test_corridor_laws_m11.py`,
  `test_penalties.py`, `test_corridor_loop.py`, `test_safety.py`, `test_windowed_analysis.py`: passed. The three new files
  with `test_models.py`, `test_residual_idm_tt.py`, `test_certificate_latent.py`, `test_closed_loop.py` and
  `test_evaluate.py` again after the line-ending repair (22:16): passed (1 skipped: the GPU test). The full quick set
  (`pytest -k "not slow"`) was not run (the machine was shared with the jobs of other sessions).
  `scripts/analysis/certificate_tt.py` with its default arm reproduces `runs/_tables/m11/certificate_tt` byte for byte.

### B'.8 Hash neutrality

- Dry runs after the implementation (19:12-19:20) of the eight queues of B'.0: byte-identical to those before
  (`cmp`; e1 175, e4 100, e4_rmax 200, e4_tt 10, e4_tt_pilot 3, m8_temporal 15, e2_windowed 15, e4_variant_d 50 jobs).
  No key was added to `configs/train.yaml`, `configs/run_corridor.yaml` or `configs/audit_stability.yaml`; the new
  model config is that of new runs only; the law files of the existing laws are not rewritten (`export_laws(only=...)`)
  and a law file gets `stateful` only for stateful members.
- `python scripts/make_tables.py` into the scratchpad (19:35): every file of `runs/_tables/m4` (20) and
  `runs/_tables/m8/e2_monotone` byte-identical to the stored ones.
- `python scripts/corridor_metrics.py compute=false tables=true` into the scratchpad (19:40): every table of
  `runs/_tables/m5` and `m8` byte-identical; `missing.txt` and `missing_corridor.txt` gain four lines, the laws of M12
  without a law file yet (gru_wmax, lstm_wmax, perl_wmax, residual_idm_tt2).
- **Found and fixed on the way (blocking):** at the committed state (v2.1.0) `corridor_metrics.py tables=true` and the
  pairs scripts stop with `unknown keys of the corridor tables: ['filter_laws']`: the design key `filter_laws` of
  `configs/corridor_metrics.yaml` (D130, read only by `figures_supplement.py`) is not a field of
  `CorridorTablesConfig`. Fix (one field appended next to `m12_laws` in `cf_stability/eval/corridor_tables.py`):
  `filter_laws: tuple = ()`, accepted and not used by the tables; the check above ran with it.
- Line endings: my first edits of LF files went through Python text mode on Windows and wrote CRLF; every file whose
  committed version uses LF was set back to LF (byte replacement; the files committed with CRLF, e.g. `trainer.py`,
  `penalties.py`, `scripts/train.py`, keep CRLF). The shared files `corridor_tables.py`, `figures_supplement.py`,
  `corridor_metrics.yaml` were among them (LF again at 19:25).

### B'.9 Commands run

```bash
python scripts/run_experiment.py configs/queue/<q>.yaml --dry-run      # q in e1 e4 e4_rmax e4_tt e4_tt_pilot m8_temporal
                                                                         # e2_windowed e4_variant_d, before and after
python scratchpad/m12_B/bench_step.py follownet_highd 40                 # step times (B'.2)
python scratchpad/m12_B/bench_step.py ngsim_i80 30
python scratchpad/m12_B/bench_pass.py; python scratchpad/m12_B/bench_gru.py
python scripts/train.py data=follownet_highd model=residual_idm_tt2 fold=0 seed=0 calibration.stability_margin=0.2 \
    certificate.enforce=true model.r_max=0.3 model.tau_z=15 train.max_epochs=2 train.steps_per_epoch=10 \
    experiment=m12_smoke paths.runs_root=<scratchpad>/smoke_runs       # smoke run, then its audit, frozen audit,
python scripts/audit_stability.py run=<smoke run> [frozen_z=-0.2]       # platoon test and certificate
python scripts/platoon_test.py run=<smoke run>; python scripts/certificate.py run=<smoke run>
python scripts/run_experiment.py configs/queue/e4_tt2_pilot.yaml        # 3 runs, 18:37-19:23
bash scratchpad/m12_B/frozen_audits.sh <run>...                         # audit_stability.py run=<run> frozen_z=<z>,
                                                                         # z in -1 -0.5 -0.2 0 0.5 1 (restartable)
python scripts/analysis/certificate_tt.py --out <scratchpad>/cert_tt_check   # the default arm: the M11 table, identical
python -m pytest tests/test_residual_idm_tt2.py tests/test_stateful_callers.py tests/test_corridor_laws_m12.py \
    tests/test_models.py ...                                             # see B'.7
CF_GPU_TESTS=1 python -m pytest tests/test_residual_idm_tt2.py -k cuda_graph
python scripts/make_tables.py paths.out_dir=<scratchpad>/m4 paths.out_dir_m8=<scratchpad>/m8      # check: identical
python scripts/corridor_metrics.py compute=false tables=true paths.out_dir=<scratchpad>/m5 \
    paths.out_dir_m8=<scratchpad>/m8                                    # check: identical but for missing.txt
python scripts/analysis/tt2_summary.py                                  # tables tt2_pilot, tt2_highd, tt2_ngsim (rerun
                                                                         # after every stage; final 22:12)
python scripts/run_experiment.py configs/queue/e4_tt2.yaml              # 10 runs, 19:31-22:05
bash scratchpad/m12_B/frozen_watch.sh                                   # frozen_audits.sh of every e4_tt2* run as soon
                                                                         # as its queue steps are done (60 audits)
python scripts/analysis/certificate_tt.py --arm tt2                     # table certificate_tt2 (22:08)
python scratchpad/m12_B/export_tt2_law.py                               # export_laws(laws.yaml, only=[residual_idm_tt2])
python scripts/check_environment.py                                     # 17 OK
python scripts/run_corridor.py scenarios=[i80_p0,i80_p1,i80_p2,us101_p0,us101_p1,us101_p2] \
    laws=[residual_idm_tt2] seeds=[0,1,2,3,4,5,6,7,8,9] workers=3       # 60 runs, 22:06-23:36
python scripts/corridor_metrics.py law=residual_idm_tt2                 # 60 macro.json (the three I-80 truths rewritten,
                                                                         # byte-identical to the committed files)
python scripts/corridor_asymmetry.py law=residual_idm_tt2 workers=2     # 60 asymmetry.json
python scripts/make_tables.py                                           # runs/_tables/m4, m8/e2_monotone: unchanged
python scripts/corridor_metrics.py compute=false tables=true            # runs/_tables/m5, m8: the rows of the new law
python scripts/analysis/m12_pairs.py                                    # table tt2_pairs
python scripts/analysis/tt2_summary.py                                  # final tt2_pilot, tt2_highd, tt2_ngsim
```

### B'.10 Run counts and durations

- Training (GPU shared with agent A' and other sessions; the main queue waited 25 min for free memory before its first
  job, 19:31-19:56): pilot 3 runs (train.py 41.0, 5.9, 9.4 min; jobs with audit, platoon and certificate 45.5, 8.6,
  14.7 min; 18:37-19:23); e4_tt2 5 runs (train.py 11.7-41.7 min, 107.7 min in all; jobs 13.8-44.0 min) and e4_tt2_ft
  5 runs (train.py 14.4-41.6 min, 121.4 min in all; jobs 16.3-43.6 min); the queue e4_tt2 ran 19:31-22:05 over 2
  workers, 0 failed. About 4.8 h of training in all (the contract estimated 4 h of GPU).
- Audits: 13 online audits (in the queues, 50-132 s) and 78 audits at frozen latents (13 runs x 6 z: the five of
  D129 and -0.2; 21-77 s each; one attempt failed with a CUDA out-of-memory error under the load of 18:50 and was run
  again; the six error files of the first, misdirected attempt: B'.6, item 11). 13 platoon tests, 13 certificates.
- Corridor: 60 runs (I-80 4.4-5.1 min, US-101 3.4-4.5 min each, 4.4 h of run time over 3 workers, 22:06-23:36); metrics
  and asymmetry of the law 18 s (the fields of the runs cached); tables 3 min.
- Smoke run `<scratchpad>/smoke_runs/m12_smoke` (2 epochs) outside the repository.
