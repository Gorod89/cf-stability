# M12 contract: A1' (the windowed-gain penalty with the maximum over the frequencies) and A2' (the two-timescale law with a persistent latent)

Drafted 20 October 2026 from the diagnoses of M11 (`docs/m11_report.md`, D128 and D129); not started. Both items
repair a specific flaw found in M11, keep everything else of the M11 design, and are judged by the rules of M11 so
that the before/after comparison is clean. Nothing of M12 enters paper/v2. Results go to `docs/m12_report.md`,
`runs/_tables/m12/` and, where a law enters the corridor, the corridor tables as further rows (design keys as in
D133). Decisions D134-D136; deviations are recorded by the agents and by the main session in `docs/decisions.md`.

Constraints as in M11 (D133): the shared `.venv` (no new packages); new configuration keys hash-neutral at their
defaults (`LATE_KEYS`, D110) so that no existing run changes its hash; the GPU shared (two training processes per
agent at most, `min_free_ram_gb` 6); restartable queues, logs in the scratchpad, tests for every new module; no AI
mentions in committed files; agents do not commit; the release script is never piped (its `pushed:` line is checked).

## D134 - A1': the maximum over the frequencies in the windowed-gain penalty (agent A')

The flaw (D128, A.9 item 8): the gain term of `windowed_gain` is the mean over the 25 audit frequencies of
relu(|G_w| - 1 + 0.02)^2, so a broad amplification of 1.1 near 1 rad/s costs about 0.002, below the feasibility
tolerance 0.01 of D89; the training declares the penalty met while the audit finds every speed unstable. D81 met
the same weakness in `linear_gain` and took the maximum.

The change: a key `train.penalty.frequency_reduce` of `configs/train.yaml` with the values `mean` (the M11 behaviour,
the hash-neutral default) and `max`; with `max` the gain term of every equilibrium is max_w relu(|G_w| - 1 + 0.02)^2
(the frequency of the largest excess; its gradient flows into that frequency only, as in `linear_gain`), the other
two terms (M_w, poles) and the existence term unchanged. With `max` an amplification of 1.1 costs (0.12)^2 = 0.0144 >
0.01, so the feasibility rule of D89 sees it. No other change of the penalty; the tests of D128 are rerun with both
values and a new test checks that `max` equals the largest per-frequency term on the linear toy law and that the
gradient reaches the right frequency. The history records the frequency of the largest excess per epoch
(`info["argmax_omega"]`) so that the migration of the amplification can be read.

Runs (restartable queues; the GPU shared with agent B'): `e2_wmax_pilot` (GRU, LSTM, PERL; highD fold 0, seed 0;
weights 0.1, 1, 10, 100 - the M11 grid plus one step up, because the maximum is a larger penalty than the mean;
`every` 4 as in M11; steps train, audit, platoon), the weights chosen by D85 (tie rule as in D128) and written into
`e2_wmax` (5 folds, seed 0); the full-history audit (`full_history_audit.py --arm-set m12`) and the long-window check
(`long_window_check.py` on the fold-0 pilot runs) as in D128; the fine-tuning `m12_ft_wmax` on NGSIM I-80 with the
penalty continued; the laws `gru_wmax`, `lstm_wmax`, `perl_wmax` (design key `wmax_laws`, handled as `windowed_laws`),
180 corridor runs on the six scenarios with seeds 0-9, metrics, asymmetry, the corridor tables, and the paired
tables `m12/wmax_pairs` (gru_wmax against gru_windowed and against gru; lstm_wmax against lstm_windowed and against
lstm; paired by scenario and seed with the builder of `ablation_pairs`).

Decision rule of H-N3' (5-fold means on highD against the E1 runs of the same architectures, as D128): confirmed
when GRU and LSTM reach a stable band share >= 0.9 at a test spacing RMSE increase <= 10 % and the long-window check
holds at the chosen weights; partial when the share is reached at a larger increase; refuted otherwise; PERL reported.
Secondary, reported with intervals and no verdict: the share of speeds with the largest windowed gain above 1.02
near 1 rad/s (0.5-2 rad/s) before and after the change; the corridor macro error and the contact rate of gru_wmax
against gru_windowed (M11: 0.132 / 0.260, contacts 270-298) and lstm_wmax against lstm_windowed (0.423 / 0.679).

## D135 - A2': the two-timescale certified law with a latent carried along the vehicle (agent B')

The flaw (D129, B.2): the latent of `residual_idm_tt` restarts at z = 0 with every 30-state window the model is
called with, so |z| <= 1 - (1 - dt / tau_z)^30 = 0.182 at tau_z = 15 s (T within +-7.7 % of T_c); the "slow" variable
is a feature of the window, not a state of the vehicle. A second limit: the margin core of E4 cannot be certified
below T = 0.92 T_c (margin 0 at T_c / 1.5), so the certificate held at 13 of the 21 latents of [-1, 1].

The change, model `residual_idm_tt2` (`cf_stability/models/residual_idm_tt2.py`, a subclass of `ResidualIDMTT`):

1. *Range.* z in [z_lo, z_hi] = [-0.2, 1.0] by z_target = z_lo + (z_hi - z_lo) sigmoid(g_phi(h_t)), T(z) = T_c
   exp(z ln 1.5) as in D129 (T from 0.92 T_c to 1.5 T_c), so that the exact certificate of D129 holds at every
   reachable z by construction; the budget of D86 is enforced over 21 values of [z_lo, z_hi] (the sweep of
   `certificate_tt.py` at the same 21 values, before and after fine-tuning: it must hold at all of them).
2. *Persistence.* The model is stateful: `stateful = True`, `initial_state(batch, device)` (z = 0), `forward(history,
   state=None) -> (acceleration, state)` with `state = None` meaning the M11 behaviour (the window-restarted latent),
   so that every existing caller keeps working. The latent is advanced once per call with the window's last step,
   z_{t+1} = z_t + (dt / tau_z)(z_target - z_t), and returned. Three callers carry the state: (a) the trainer: a
   teacher-forced pass of the latent over the event from its first window to the start of every rollout (gradients
   kept, no detach; the pass runs once per event of the batch and is shared by the rollouts starting inside it), then
   the closed-loop rollout carries it; (b) the audit, the gain and long-window rollouts and the platoon test: the state
   initialised at the warm-up on the equilibrium and carried through the rollout (the frozen-z audits of D129 stay as
   they are); (c) the corridor law (`ModelLaw`): one state per vehicle, created at its insertion and carried with its
   history window. A test for each caller: the persistent latent of a constant input converges to z_target with the
   time constant tau_z (not to the window bound 0.182); a rollout with `state=None` reproduces the M11 model bit for
   bit; the corridor loop on the synthetic scenario of `tests/test_corridor_loop.py` carries the state (two calls with
   and without persistence differ, the hash of the run config records the model's `stateful`).
3. *Everything else as D129*: the residual with z as a fourth input, the same derivative budget at every z, tau_z in
   {15, 30, 60} s in the pilot chosen by the rule of D129, `frozen_z`, the one-layer GRU (hidden 16) of the latent.

Runs: `e4_tt2_pilot` (highD fold 0, seed 0, the three tau_z; the latent's range on the test events reported: 1 %,
median, 99 % quantiles of z, which must now spread beyond 0.182), `e4_tt2` (5 folds) and `e4_tt2_ft` (NGSIM I-80, 5
folds, the certificate rechecked), the audits at the frozen z of D129 and online, the 21-z exact certificate before
and after fine-tuning (table `m12/certificate_tt2`), the law `residual_idm_tt2` (design key `m12_laws`, handled as
`m11_laws`), 60 corridor runs on the six scenarios with seeds 0-9, metrics, asymmetry, the corridor tables, the
paired tables `m12/tt2_pairs` (tt2 against E and against tt, paired by scenario and seed).

Decision rule of H-N2' (5-fold means on highD): confirmed when the test spacing RMSE is <= 1.05 x the GRU's E1 RMSE,
every frozen-z audit and the online audit have 100 % stable equilibria and the certificate holds at all 21 z of
[z_lo, z_hi] after fine-tuning; partial when the RMSE is <= 1.15 x; refuted otherwise. Reported with intervals and no
verdict: the RMSE against E4 (M11: 0.935) and against tt (4.218 m); the latent's spread; the corridor macro error
against E (0.125 / 0.113) and against tt (0.118 / 0.090).

## D136 - reporting, ownership, compute

Each agent writes its section of `docs/m12_report.md` after every step (as M11 taught: incrementally, so that a
dropped connection loses nothing) and leaves its tables in `runs/_tables/m12/`; the main session assembles the
verdicts, the decision log and the release (v2.2.0; the manuscript's tables stay those of the tag v2.0.2). Ownership:
A' - `cf_stability/stability/penalties.py` (the key and the reduction), `configs/train.yaml` (one key), the queues
`e2_wmax*`, `m12_ft_wmax`, the law entries `*_wmax`, the block `wmax_laws` of `configs/corridor_metrics.yaml` and of
`corridor_tables.py`, `scripts/analysis/windowed_gain.py` (an `--arm` option for the m12 tables) and
`full_history_audit.py` (`--arm-set m12`); B' - `cf_stability/models/residual_idm_tt2.py`, the stateful interface of
`cf_stability/models/base.py` (additive), the three callers (`cf_stability/train/`, `cf_stability/stability/audit.py`
and `penalties.py` rollouts - coordinated with A' by appending only -, the platoon test, `cf_stability/corridor/laws.py`
ModelLaw), the certificate extension, the queues `e4_tt2*`, the block `m12_laws`. Shared files are edited by appending
only. The supplement grids leave the new laws out (`EXCLUDED_LAW_SETS` gets `wmax_laws` and `m12_laws`).

Compute (from the M11 timings): A' about 12 h of GPU (pilot 12 runs of 10-40 min, 12 fold runs, 15 fine-tuning runs)
plus 1.5 h of corridor runs; B' about 4 h of GPU (13 runs of 6-12 min) plus 0.7 h of corridor runs and the
implementation of the stateful path, which is the main risk of M12: if the trainer's event-level latent pass cannot be
done within the existing loader, the fallback is a latent carried only across the windows of one rollout (5 s) plus a
longer warm-up window, recorded as a deviation.
